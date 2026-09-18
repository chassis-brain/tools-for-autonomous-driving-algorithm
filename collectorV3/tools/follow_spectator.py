import time
import carla


HOST = "127.0.0.1"
PORT = 23000

# ===== View parameters =====
HEIGHT = 28.0       # camera height above vehicle, meters
BACK = 8.0          # distance behind vehicle, meters
PITCH = -70.0       # -90 = straight top-down
# ===========================


client = carla.Client(HOST, PORT)
client.set_timeout(10.0)

world = client.get_world()
spectator = world.get_spectator()

print("[Spectator] Connected to CARLA.")
print("[Spectator] Waiting for ego vehicle...")


def find_ego():
    vehicles = world.get_actors().filter("vehicle.*")

    # Bench2Drive / Leaderboard normally uses role_name=hero.
    for vehicle in vehicles:
        role = vehicle.attributes.get("role_name", "")
        if role == "hero":
            return vehicle

    # Fallbacks for custom setups.
    for vehicle in vehicles:
        role = vehicle.attributes.get("role_name", "")
        if role in ("ego", "ego_vehicle"):
            return vehicle

    return None


ego = None

while ego is None:
    ego = find_ego()
    if ego is None:
        time.sleep(0.2)

print(
    "[Spectator] Following ego:",
    ego.id,
    ego.type_id,
    ego.attributes.get("role_name", "")
)

while True:
    try:
        snapshot = world.wait_for_tick(10.0)

        actor_snapshot = snapshot.find(ego.id)

        # Route ended / actor was destroyed.
        if actor_snapshot is None:
            print("[Spectator] Ego disappeared. Waiting for next ego...")
            ego = None

            while ego is None:
                ego = find_ego()
                if ego is None:
                    time.sleep(0.2)

            print("[Spectator] Following new ego:", ego.id)
            continue

        transform = actor_snapshot.get_transform()

        # Vehicle's forward direction.
        forward = transform.get_forward_vector()

        # Camera = behind + above ego.
        camera_location = carla.Location(
            x=transform.location.x - BACK * forward.x,
            y=transform.location.y - BACK * forward.y,
            z=transform.location.z + HEIGHT,
        )

        # Same heading as vehicle, looking steeply downward.
        camera_rotation = carla.Rotation(
            pitch=PITCH,
            yaw=transform.rotation.yaw,
            roll=0.0,
        )

        spectator.set_transform(
            carla.Transform(camera_location, camera_rotation)
        )

    except KeyboardInterrupt:
        print("\n[Spectator] Stopped.")
        break
    except RuntimeError:
        # World may reload between routes.
        time.sleep(0.2)
        world = client.get_world()
        spectator = world.get_spectator()
        ego = None

        while ego is None:
            ego = find_ego()
            if ego is None:
                time.sleep(0.2)
