from carla_objects import *


try:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)

    world = World(client, "road_race", (640, 480))
    while not world.is_done:
        runner = Vehicle(world, "runner", bp_filter="vehicle.tesla.model3")
        while not runner.has_collided and not world.is_done:
            events = world.key_handler()

            runner.key_handler(events)
            if runner.auto_pilot:
                d = runner.distance_left - runner.distance_right
                runner.throttle = 0.3
                runner.steer = max(-1, min(1, -d * 0.2))
            runner.action()

            world.render_image(runner.rgb_image)
            world.render_text("%.1ffps" % world.clock.get_fps(), (10, 10))
            world.render_text("%.2fkm/h" % runner.speed_kmh(), (10, 30))
            world.render_text("Throttle: %.2f" % runner.throttle, (10, 50))
            world.render_text("Brake: %.2f" % runner.brake, (10, 70))
            world.render_text("Steer: %.2f" % runner.steer, (10, 90))
            world.render_text("Reverse: %d" % runner.reverse, (10, 110))
            world.render_text("L: %.2f, R: %.2f" % (runner.distance_left, runner.distance_right), (10, 130))
            world.redraw_display()
        runner.destroy()
except Exception as e:
    print(e)
finally:
    print("done.")
