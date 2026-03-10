# nodes/log.py
def log_node_info(node_name, message):
    print(f"[INFO][{node_name}] {message}")

def log_node_warn(node_name, message, msg_color="YELLOW"):
    print(f"[WARN][{node_name}] {message}")
