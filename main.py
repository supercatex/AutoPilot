from carla_objects import *


try:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)

    world = World(client, "road_race")
    while not world.is_done:
        runner = Vehicle(world, "runner", bp_filter="vehicle.tesla.model3")
        while not runner.has_collided and not world.is_done:
            events = world.key_handler()
            runner.key_handler(events)

            runner.action()

            world.render_image(runner.rgb_image)
            world.render_text("%.1ffps" % world.clock.get_fps(), (10, 10))
            world.render_text("%.2fkm/h" % runner.speed_kmh(), (10, 50))
            world.redraw_display()
        runner.destroy()
except Exception as e:
    print(e)
finally:
    print("done.")
