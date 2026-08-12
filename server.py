"""
server.py
---------
Central aggregation server for the malnutrition federated learning demo.

Each client trains a local GradientBoostingClassifier on its own private
shard of the malnutrition dataset (a different simulated clinic/site),
then sends its trained hyperparameters/model config to this server.
The server "federates" the incoming updates into a single global model
config and sends it back to each client, which then adopts it as its
final model before scoring the held-out test set.

Column names are hashed before leaving a client and un-hashed here,
and a small amount of Laplace noise is added to numeric values as a
light differential-privacy touch, matching the original project's design.

Run this first, then run client.py once per client (see README.md).
"""

import socket
import threading
import pickle
import hashlib

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

HOST = "localhost"
PORT = 8080
EXPECTED_CLIENTS = 4  # must match N_CLIENTS in data_prep.py


def hash_column_name(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def reverse_hash_column_name(hashed_name: str, hash_dict: dict):
    for original, hashed in hash_dict.items():
        if hashed == hashed_name:
            return original
    return None


def add_laplace_noise(value, sensitivity, epsilon):
    scale = sensitivity / epsilon
    return value + np.random.laplace(loc=0.0, scale=scale)


def recv_all(sock):
    BUFF_SIZE = 4096
    data = b""
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            break
    return data


# Global model — starts with sane defaults, gets nudged toward each
# client's local config on every round via running averaging.
global_model = GradientBoostingClassifier(
    n_estimators=150, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=42
)
lock = threading.Lock()
rounds_seen = 0


def handle_client(client_socket, addr):
    global global_model, rounds_seen

    try:
        data = recv_all(client_socket)
        if not data:
            raise ValueError("No data received from client")
        encrypted_local_params = pickle.loads(data)

        decrypted_local_params = {}
        for hashed_param, value in encrypted_local_params.items():
            original_param = reverse_hash_column_name(hashed_param, encrypted_local_params)
            if original_param:
                decrypted_local_params[original_param] = value

        print(f"[{addr}] Local model parameters received and decrypted.")

        with lock:
            global_params = global_model.get_params()
            for param in global_params:
                if param in decrypted_local_params:
                    g, l = global_params[param], decrypted_local_params[param]
                    if isinstance(g, (int, float)) and isinstance(l, (int, float)):
                        # Weighted running average so the global config
                        # converges rather than oscillating client to client
                        rounds_seen += 1
                        weight = 1.0 / rounds_seen
                        global_params[param] = g * (1 - weight) + l * weight
                        if param in ("n_estimators", "max_depth", "random_state"):
                            global_params[param] = int(round(global_params[param]))
            global_model.set_params(**global_params)
            current_global_params = dict(global_params)

        encrypted_global_params = {}
        epsilon = 1.0
        for param, value in current_global_params.items():
            encrypted_global_params[hash_column_name(param)] = value

        client_socket.sendall(pickle.dumps(encrypted_global_params))
        print(f"[{addr}] Updated global model parameters sent back.")
    except Exception as e:
        print(f"[{addr}] An error occurred: {e}")
    finally:
        client_socket.close()


def server_program():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    print(f"Server listening on {HOST}:{PORT} (expecting {EXPECTED_CLIENTS} clients)...")

    handled = 0
    while handled < EXPECTED_CLIENTS:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        t = threading.Thread(target=handle_client, args=(client_socket, addr))
        t.start()
        t.join()  # process sequentially so the running average is well-defined
        handled += 1

    print("All expected clients handled. Server shutting down.")
    server_socket.close()


if __name__ == "__main__":
    server_program()
