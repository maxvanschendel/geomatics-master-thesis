import paramiko
from sty import fg


class RemotePiZeroDeviceException(Exception):
    pass


class RemotePiZeroDevice:
    """Remote device connection which can be used to automate network connection and data capture."""

    # Where the scripts for the Raspberry Pi Zero Device are stored.
    repo_path = "~/Dev/geomatics-master-thesis/src/main/agents/pi_zero_bodycam"

    # Bash commands for connecting to a network, starting and stopping data capture.
    wifi_connect_cmd = f"sudo bash {repo_path}/connect_wifi.sh"
    start_capture_cmd = f"sudo python3 {repo_path}/pizero_capture_client.py 192.168.1.208 8000 640 480"
    stop_capture_cmd = "pkill -9 -f pizero_capture_client.py"
    calibrate_cmd = f"sudo python3 {repo_path}/camera_calibration.py 192.168.1.208 8000 640 480 20"

    # Colour used for printing remote device stdout and stderr.
    remote_std_color = fg.yellow

    def __init__(self, host: str, user: str, pwd: str, verbose: bool = False):
        # Connection details.
        self.hostname: str = host
        self.username: str = user
        self.password: str = pwd

        # Print stdin, stdout and stderror for executed commands.
        self.verbose: bool = verbose

        # SSH connection and how often it is validated.m
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.connected = False

    def __str__(self):
        return f"host: {self.hostname} | user: {self.username}"

    def ssh_connect(self):
        """Connect to device over SSH and keep checking if the connection is alive."""

        print(f"Connecting to remote device | {str(self)}")
        if self.connected:
            raise RemotePiZeroDeviceException("Already connected to remote device.")

        self.ssh.connect(self.hostname, username=self.username, password=self.password)
        if not self.ssh_is_active():
            raise RemotePiZeroDeviceException("Failed to connect to remote device.")

        self.connected = True

    def ssh_disconnect(self):
        """Close connection to remove device."""

        print(f"Disconnecting from remote device | {str(self)}")
        if not self.connected:
            raise RemotePiZeroDeviceException("Can't disconnect because device is not connected.")

        self.ssh.close()
        self.connected = False

    def ssh_is_active(self):
        """Returns whether the SSH connection is still active."""

        return self.ssh.get_transport().is_active()

    def ssh_validate(self):
        """Raise an exception is the SSH connection is not active."""

        if not self.ssh_is_active():
            raise RemotePiZeroDeviceException("SSH connection to device not active.")

    def exec(self, command: str):
        """Execute bash command on remote device."""

        self.ssh_validate()
        ssh_stdin, ssh_stdout, ssh_stderr = self.ssh.exec_command(command, get_pty=True)

        if self.verbose:
            self.print_remote_std(ssh_stderr)
            self.print_remote_std(ssh_stdout)

    def print_remote_std(self, std):
        while not std.channel.exit_status_ready():
            out = std.channel.recv(1024).decode('ascii').strip("\n")
            print(self.remote_std_color + out + fg.rs) if len(out) > 1 else None

    def connect_network(self):
        print(f"Connecting to network")
        self.exec(self.wifi_connect_cmd)

    def calibrate(self):
        print(f"Calibrating")
        self.exec(self.calibrate_cmd)

    def start_capture(self):
        print(f"Starting data capture")
        self.exec(self.start_capture_cmd)

    def stop_capture(self):
        print(f"Stopping data capture")
        self.exec(self.stop_capture_cmd)


if __name__ == "__main__":
    hostname = "raspberrypi.local"
    username = "pi"
    password = "maxvanschendel"
    verbose = True

    remote_pi = RemotePiZeroDevice(hostname, username, password, verbose)
    remote_pi.ssh_connect()
    remote_pi.connect_network()

    command_actions = {'1': remote_pi.start_capture, '2': remote_pi.calibrate}
    command = input("Select action:\n1) Stream video\n2) Calibrate\n>")

    if command in command_actions.keys():
        command_actions[command]()
    else:
        print("Invalid action")

    remote_pi.ssh_disconnect()
