from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower

config = SO101FollowerConfig(
    port="COM6",
    id="DI_VLA_FOLLOWER",
)

follower = SO101Follower(config)
follower.connect(calibrate=False)
follower.calibrate()
follower.disconnect()