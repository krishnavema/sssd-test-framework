from __future__ import annotations

import re
import tempfile
import time

from pytest_mh import BackupTopologyController
from pytest_mh.conn import ProcessResult

from .config import SSSDMultihostConfig
from .hosts.ad import ADHost
from .hosts.client import ClientHost
from .hosts.ipa import IPAHost
from .hosts.kdc import KDCHost
from .hosts.keycloak import KeycloakHost
from .hosts.ldap import LDAPHost
from .hosts.samba import SambaHost
from .misc.ssh import retry_command

__all__ = [
    "LDAPTopologyController",
    "LDAPKRB5TopologyController",
    "IPATopologyController",
    "ADTopologyController",
    "SambaTopologyController",
    "IPATrustADTopologyController",
    "IPATrustSambaTopologyController",
    "KeycloakTopologyController",
]


class ProvisionedBackupTopologyController(BackupTopologyController[SSSDMultihostConfig]):
    """
    Provide basic restore functionality for topologies.
    """

    _last_refresh_time: float | None = None

    def __init__(self) -> None:
        super().__init__()

        self.provisioned: bool = False

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.provisioned = self.name in self.multihost.provisioned_topologies

    # Per-host connect timeout left on the session so pytest-mh's lazy reconnect
    # (which happens on the next command after we disconnect below) is not stuck
    # at the 1s timeout pytest-mh lowers it to after the first connect. Long
    # enough for a slow AWS/ITUP handshake.
    _REFRESH_CONNECT_TIMEOUT: int = 15

    # Hard wall-clock budget for a single refresh pass across ALL hosts. Kept as
    # a backstop; with disconnect-only refresh a pass is effectively instant.
    _REFRESH_TIME_BUDGET: int = 90

    # Host types we must NEVER proactively disconnect. The AD domain controller
    # is a Windows host reached over Windows OpenSSH (user "Administrator"): it
    # does not tolerate having its single persistent session force-closed and
    # then reconnected, and once wedged it never re-accepts a TCP connection for
    # the rest of the run. That wedges every subsequent AD test (all of which
    # need the DC at setup to create users). These hosts also do comparatively
    # few operations, so they are not where libssh channel exhaustion occurs.
    # Samba is included for the same Windows-like fragility reasons.
    _REFRESH_SKIP_TYPES: tuple = (ADHost, SambaHost)

    def _set_connect_timeout(self, host, seconds: int) -> None:
        """
        Set the underlying libssh connect timeout on a host's session.

        pytest-mh's ``SSHClient`` reuses a single ``LibsshSession`` and, after
        the first successful connect, lowers its timeout to 1 second (see
        ``SSHClient.connect``). Because the session object is reused, any later
        (re)connect inherits that 1s timeout, so the handshake to slower hosts
        (e.g. AWS/ITUP VMs) fails with "ssh connect failed: Timeout
        connecting". We set a sane timeout both before we reconnect here AND so
        that pytest-mh's own lazy reconnect (used when we give up below) is not
        left stuck at 1s. pytest-mh drops it back to 1s once a connect succeeds,
        so steady-state behaviour is unaffected.
        """
        session = getattr(getattr(host, "conn", None), "_SSHClient__conn", None)
        if session is None:
            return
        try:
            session.set_ssh_options("timeout", seconds)
        except Exception as e:
            self.logger.warning(f"Could not set connect timeout for {host.hostname}: {e}")

    def _refresh_connections(self) -> None:
        """
        Disconnect stale SSH sessions so libssh channels that accumulate over
        hundreds of test-driven open/close cycles are freed ("Failed to
        open_session: [-2]" channel exhaustion). We do NOT eagerly reconnect:
        pytest-mh reconnects lazily on the next command against each host, which
        is both cheaper and far safer.

        Safety properties (why this cannot repeat the ~24 min stall + cascade,
        nor the AD-DC-wedge cascade, that earlier versions caused):

        * Throttled: runs at most once every 120s (class-wide).
        * Disconnect-only: we never hammer a host with an immediate reconnect.
          The eager disconnect+reconnect wedged the Windows AD DC (it stopped
          accepting connections for the rest of the run) and stalled teardown
          15s x N hosts. Lazy reconnect avoids both.
        * Skips fragile hosts: the AD/Samba (Windows) hosts are never touched
          (see ``_REFRESH_SKIP_TYPES``); they are not where channel exhaustion
          occurs and cannot survive a forced session close.
        * Non-fatal: every failure is swallowed and a sane connect timeout is
          left on the session for pytest-mh's lazy reconnect.
        """
        now = time.monotonic()
        if ProvisionedBackupTopologyController._last_refresh_time is None:
            ProvisionedBackupTopologyController._last_refresh_time = now
            return

        elapsed = now - ProvisionedBackupTopologyController._last_refresh_time
        if elapsed < 120:
            return

        ProvisionedBackupTopologyController._last_refresh_time = now

        all_hosts = []
        for domain in self.multihost.domains:
            all_hosts.extend(domain.hosts)

        self.logger.info(
            f"Refreshing SSH connections for {len(all_hosts)} hosts after {int(elapsed)}s "
            f"to prevent channel exhaustion"
        )

        deadline = now + self._REFRESH_TIME_BUDGET
        for host in all_hosts:
            if time.monotonic() >= deadline:
                # Out of budget (should not happen with disconnect-only, kept as
                # a backstop). Leave the rest for lazy reconnection.
                self.logger.warning(
                    f"Refresh time budget ({self._REFRESH_TIME_BUDGET}s) exhausted; "
                    f"skipping remaining hosts, relying on lazy reconnection"
                )
                break

            # Never proactively disconnect the Windows AD/Samba DC: forcing its
            # session closed permanently wedges its SSH server and cascades into
            # every subsequent AD test failing at setup.
            if isinstance(host, self._REFRESH_SKIP_TYPES):
                self.logger.info(f"Skipping connection refresh for fragile host {host.hostname}")
                continue

            self.logger.info(f"Refreshing SSH connection to {host.hostname}")
            # Disconnect only: this frees the leaked libssh channels. We do NOT
            # eagerly reconnect -- pytest-mh reconnects lazily on the next
            # command, which avoids both the teardown stall (15s x N hosts) and
            # the reconnect storm that wedged hosts. Leave a sane connect timeout
            # so that lazy reconnect is not stuck at pytest-mh's 1s default.
            try:
                host.conn.disconnect()
            except Exception as e:
                self.logger.warning(f"Disconnect failed for {host.hostname}: {e}")
            self._set_connect_timeout(host, self._REFRESH_CONNECT_TIMEOUT)

    def topology_setup(self, *args, **kwargs) -> None:
        self._refresh_connections()

        if self.provisioned:
            self.logger.info(f"Topology '{self.name}' is already provisioned")
            return

        super().topology_setup(*args, **kwargs)

    def topology_teardown(self, *args, **kwargs) -> None:
        self._refresh_connections()

        if self.provisioned:
            return

        super().topology_teardown(*args, **kwargs)

    def teardown(self) -> None:
        self._refresh_connections()

        if self.provisioned:
            self.restore_vanilla()
            return

        super().teardown()

    def join_domain(self, client: ClientHost, provider: IPAHost | ADHost | SambaHost):
        """
        Helper method for joining domains
        """
        self.logger.info(f"Enrolling {client.hostname} into {provider.domain}")

        # Remove any existing Kerberos configuration and keytab
        client.fs.rm("/etc/krb5.conf")
        client.fs.rm("/etc/krb5.keytab")

        # First check if joined.  If so, leave.
        result = client.conn.exec(["realm", "list"])
        self.logger.info(f"REALM_LIST STDOUT: \n{result.stdout}")
        self.logger.info(f"REALM_LIST STDERR: \n{result.stderr}")
        pattern = rf"^{re.escape(provider.domain)}$"
        if re.search(pattern, result.stdout, re.MULTILINE):
            self.logger.info(f"Found {provider.domain} in realm list.  Leaving.")
            result = client.conn.exec(
                ["realm", "leave", provider.domain], input=provider.adminpw, raise_on_error=False
            )
            self.logger.info(f"Optional realm leave failed...continuing to join {provider.domain}.")

        if isinstance(provider, (IPAHost)):
            # Backup ipa-client-install files
            client.fs.backup("/etc/ipa")
            client.fs.backup("/var/lib/ipa-client")

        # Join provider domain
        result = client.conn.exec(["realm", "join", provider.domain], input=provider.adminpw, raise_on_error=False)
        if result.rc != 0:
            self.logger.info(f"Running realm join failed with:\n{result.stdout}\n{result.stderr}")
            self.logger.info("Trying uninstall and join again.")
            if isinstance(provider, (IPAHost)):
                client.conn.exec(["ipa-client-install", "--uninstall", "-U"])
            else:
                client.conn.exec(["realm", "leave", "--unattended", provider.domain], input=provider.adminpw)
            client.conn.exec(["realm", "join", provider.domain], input=provider.adminpw)


class ClientTopologyController(ProvisionedBackupTopologyController):
    """
    Client Topology Controller.
    """

    pass


class LDAPTopologyController(ProvisionedBackupTopologyController):
    """
    LDAP Topology Controller.
    """

    pass


class LDAPKRB5TopologyController(LDAPTopologyController):
    """
    Bare LDAP + KDC topology (client + LDAP + KDC, **no NFS**).

    Central place for **LDAP + Kerberos** multihost setup beyond what
    :class:`LDAPTopologyController` does. Today this provisions the client's host
    principal keytab for GSSAPI; additional steps can be added here as LDAP/KRB5
    tests grow.

    Current setup:

    * Ensure ``host/<client fqdn>`` exists in the KDC database
    * Export it to a keytab on the KDC, copy to the client as ``/etc/krb5.keytab``
    * Set ``ldap_krb5_keytab`` on the LDAP provider client defaults (unless already set)

    """

    _KDC_KEYTAB_STAGING = "/root/.sssd-test-framework-host.keytab"
    _CLIENT_KEYTAB_PATH = "/etc/krb5.keytab"

    @staticmethod
    def _kadmin_local(kdc: KDCHost, command: str) -> ProcessResult:
        result = kdc.conn.exec(["kadmin.local", "-q", command])
        if len(result.stdout_lines) > 1:
            result.stdout_lines = result.stdout_lines[1:]
            result.stdout = "\n".join(result.stdout_lines)
        return result

    def _ensure_host_principal(self, kdc: KDCHost, principal: str) -> None:
        getp = self._kadmin_local(kdc, f'getprinc "{principal}"')
        # kadmin.local often returns exit code 0 even when the principal is missing
        # (error text only on stderr); do not rely on rc alone.
        combined = ((getp.stdout or "") + (getp.stderr or "")).lower()
        if "does not exist" in combined:
            self._kadmin_local(kdc, f'addprinc -randkey "{principal}"')
            return
        if getp.rc != 0:
            self._kadmin_local(kdc, f'addprinc -randkey "{principal}"')

    def _provision_client_host_keytab(self, client: ClientHost, ldap: LDAPHost, kdc: KDCHost) -> None:
        principal = f"host/{client.hostname}"
        self.logger.info(f"Provisioning host keytab for {principal} -> {self._CLIENT_KEYTAB_PATH}")

        self._ensure_host_principal(kdc, principal)

        kdc.conn.run(f"rm -f {self._KDC_KEYTAB_STAGING}", raise_on_error=False)
        self._kadmin_local(kdc, f'ktadd -k {self._KDC_KEYTAB_STAGING} -norandkey "{principal}"')

        with tempfile.NamedTemporaryFile() as tmp:
            kdc.fs.download(self._KDC_KEYTAB_STAGING, tmp.name)
            client.fs.upload(tmp.name, self._CLIENT_KEYTAB_PATH)

        kdc.conn.run(f"rm -f {self._KDC_KEYTAB_STAGING}", raise_on_error=False)
        client.conn.run(
            f"chmod 600 {self._CLIENT_KEYTAB_PATH} && chown root:root {self._CLIENT_KEYTAB_PATH}",
            raise_on_error=False,
        )

        ldap.client.setdefault("ldap_krb5_keytab", self._CLIENT_KEYTAB_PATH)

    @BackupTopologyController.restore_vanilla_on_error
    def topology_setup(
        self,
        client: ClientHost,
        ldap: LDAPHost,
        provider: LDAPHost,
        kdc: KDCHost,
    ) -> None:
        if self.provisioned:
            self.logger.info(f"Topology '{self.name}' is already provisioned")
            return

        self._provision_client_host_keytab(client, ldap, kdc)
        super().topology_setup()


class IPATopologyController(ProvisionedBackupTopologyController):
    """
    IPA Topology Controller.
    """

    @BackupTopologyController.restore_vanilla_on_error
    def topology_setup(self, client: ClientHost, ipa: IPAHost) -> None:
        short_hostname = client.conn.run("hostname").stdout.split(".")[0].strip()
        hostname = f"{short_hostname}.{ipa.domain}"
        client.fs.backup("/etc/hostname")
        client.fs.backup("/etc/hosts")
        client.conn.run(f"echo {hostname} > /etc/hostname")
        client.fs.write("/etc/hosts", client.fs.read("/etc/hosts").replace("client.test", hostname))

        # Change client hostname to match the domain
        self.logger.info(f"Changing hostname to {hostname}")
        client.conn.run(f"hostname {hostname}")

        client.fs.backup("/etc/resolv.conf")

        if self.provisioned:
            self.logger.info(f"Topology '{self.name}' is already provisioned")
            return

        # Join IPA domain
        self.join_domain(client, ipa)

        # Backup so we can restore to this state after each test
        super().topology_setup()


class ADTopologyController(ProvisionedBackupTopologyController):
    """
    AD Topology Controller.
    """

    @BackupTopologyController.restore_vanilla_on_error
    def topology_setup(self, client: ClientHost, provider: ADHost | SambaHost) -> None:
        short_hostname = client.conn.run("hostname").stdout.split(".")[0].strip()
        hostname = f"{short_hostname}.{provider.domain}"
        client.fs.backup("/etc/hostname")
        client.conn.run(f"echo {hostname} > /etc/hostname")

        # Change client hostname to match the domain
        self.logger.info(f"Changing hostname to {hostname}")
        client.conn.run(f"hostname {hostname}")

        # Workaround, downstream CI resets resolv.conf pointing to invalid servers.
        if isinstance(provider, SambaHost):
            if "127.0.0.1" not in provider.fs.read("/etc/resolv.conf"):
                provider.fs.backup("/etc/resolv.conf")
                provider.fs.write("/etc/resolv.conf", f"search {provider.domain}\nnameserver 127.0.0.1\n\n")

        if self.provisioned:
            self.logger.info(f"Topology '{self.name}' is already provisioned")
            return

        # Join AD/Samba domain
        self.join_domain(client, provider)

        # Backup so we can restore to this state after each test
        super().topology_setup()


class SambaTopologyController(ADTopologyController):
    """
    Samba Topology Controller.
    """

    pass


class IPATrustADTopologyController(ProvisionedBackupTopologyController):
    """
    IPA trust AD Topology Controller.
    """

    @BackupTopologyController.restore_vanilla_on_error
    def topology_setup(self, client: ClientHost, ipa: IPAHost, trusted: ADHost | SambaHost) -> None:
        short_hostname = client.conn.run("hostname").stdout.split(".")[0].strip()
        hostname = f"{short_hostname}.{ipa.domain}"
        client.fs.backup("/etc/hostname")
        client.conn.run(f"echo {hostname} > /etc/hostname")

        # Change client hostname to match the domain
        self.logger.info(f"Changing hostname to {hostname}")
        client.conn.run(f"hostname {hostname}")

        if self.provisioned:
            self.logger.info(f"Topology '{self.name}' is already provisioned")
            return

        # Create trust
        self.logger.info(f"Establishing trust between {ipa.domain} and {trusted.domain}")
        ipa.kinit()
        self.trust_add(ipa, trusted)

        # Do not enroll client into IPA domain if it is already joined
        if "ipa" not in self.multihost.provisioned_topologies:
            # Join IPA domain
            self.join_domain(client, ipa)

        # Backup so we can restore to this state after each test
        super().topology_setup()

    # If this command is run on freshly started containers, it is possible the IPA is not yet
    # fully ready to create the trust. It takes a while for it to start working.
    @retry_command(max_retries=20, delay=5, match_stderr='CIFS server communication error: code "3221225581"')
    def trust_add(self, ipa: IPAHost, trusted: ADHost | SambaHost) -> ProcessResult:
        return ipa.conn.exec(
            ["ipa", "trust-add", trusted.domain, "--admin", "Administrator", "--password"], input=trusted.adminpw
        )


class IPATrustSambaTopologyController(IPATrustADTopologyController):
    """
    IPA trust Samba Topology Controller.
    """

    pass


class KeycloakTopologyController(ProvisionedBackupTopologyController):
    """
    Keycloak Topology Controller.
    """

    @BackupTopologyController.restore_vanilla_on_error
    def topology_setup(self, client: ClientHost, keycloak: KeycloakHost) -> None:
        if self.provisioned:
            self.logger.info(f"Topology '{self.name}' is already provisioned")
            return

        self.logger.info(f"Enrolling {client.hostname} into {keycloak.hostname} by creating an IdP client")

        # Create an IdP client
        keycloak.kclogin()
        keycloak.conn.run(
            "/opt/keycloak/bin/kcadm.sh create clients -r master "
            '-b \'{"clientId": "myclient", "clientAuthenticatorType": "client-secret", '
            '"secret": "ClientSecret123", "serviceAccountsEnabled": true, '
            '"attributes": {"oauth2.device.authorization.grant.enabled": "true"}}\' '
        )
        keycloak.conn.run(
            "/opt/keycloak/bin/kcadm.sh add-roles -r master "
            "--cclientid account --rolename view-groups --uusername service-account-myclient"
        )
        keycloak.conn.run(
            "/opt/keycloak/bin/kcadm.sh add-roles -r master "
            "--cclientid master-realm --rolename view-users --uusername service-account-myclient"
        )
        keycloak.conn.run(
            "/opt/keycloak/bin/kcadm.sh add-roles -r master "
            "--cclientid master-realm --rolename query-users --uusername service-account-myclient"
        )
        keycloak.conn.run(
            "/opt/keycloak/bin/kcadm.sh add-roles -r master "
            "--cclientid master-realm --rolename query-groups --uusername service-account-myclient"
        )

        # Backup so we can restore to this state after each test
        super().topology_setup()


class GDMTopologyController(ProvisionedBackupTopologyController):
    """
    GDM EIdP Topology Controller.
    """

    @BackupTopologyController.restore_vanilla_on_error
    def topology_setup(self, client: ClientHost, ipa: IPAHost, keycloak: KeycloakHost) -> None:
        short_hostname = client.conn.run("hostname").stdout.split(".")[0].strip()
        hostname = f"{short_hostname}.{keycloak.domain}"

        # Change client hostname to match the domain
        self.logger.info(f"Changing hostname to {hostname}")
        client.conn.run(f"hostname {hostname}")

        if "gdm" not in client.features or not client.features["gdm"]:
            self.logger.info(f"Topology '{self.name}' setup skipped because gdm feature not found on client")
            return

        if self.provisioned:
            self.logger.info(f"Topology '{self.name}' is already provisioned")
            return

        self.logger.info(f"Enrolling IPA server {ipa.hostname} into {keycloak.hostname} by creating an IdP client")

        # Join IPA domain
        self.join_domain(client, ipa)

        # Create an IdP client
        keycloak.kclogin()
        keycloak.conn.run(
            "/opt/keycloak/bin/kcadm.sh create clients -r master "
            '-b \'{"clientId": "ipa_oidc_client", "clientAuthenticatorType": "client-secret", '
            '"secret": "IPA_Secret123", "serviceAccountsEnabled": true, '
            '"redirectUris" : [ "https://ipa-ca.ipa.test/ipa/idp/*" ], '
            '"webOrigins" : [ "https://ipa-ca.ipa.test" ],'
            '"attributes": {"oauth2.device.authorization.grant.enabled": "true"}}\' '
        )

        ipa.kinit()
        ipa.conn.run(
            f"ipa idp-add keycloak --provider keycloak --base-url {keycloak.hostname}:8443/auth "
            "--org master --client-id ipa_oidc_client --secret",
            input="IPA_Secret123",
        )

        # Backup so we can restore to this state after each test
        super().topology_setup()

    def topology_teardown(self, client: ClientHost, ipa: IPAHost, keycloak: KeycloakHost) -> None:
        if "gdm" not in client.features or not client.features["gdm"]:
            self.logger.info(f"Topology '{self.name}' teardown skipped because gdm feature not found on client")
            return

        self.logger.info(f"Un-enrolling IPA server {ipa.hostname} from {keycloak.hostname} by deleting the IdP client")
        keycloak.kclogin()
        keycloak.conn.run(
            "ID=$(/opt/keycloak/bin/kcadm.sh get clients -q clientId=ipa_oidc_client --fields=id|jq -r '.[0].id'); "
            "/opt/keycloak/bin/kcadm.sh delete clients/$ID"
        )

        ipa.kinit()
        ipa.conn.run("ipa idp-del keycloak")

        super().topology_teardown()
