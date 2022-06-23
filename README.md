4dm_webgl_viewer is a tool meant to be a viewer and perhaps eventually a world editor for the game 4D Miner by Mashpoe.

== How to run it

1. download or clone the source code to your PC
2. Run a webserver with document root inside the folder for this project.
    On Windows 10, one easy way to do this is to have python installed, with http.server module available,
    and use a command prompt to cd into this project's directory and run `python -m http.server`.
    Then it's easy to point your browser to http://localhost:8000 to open this application.

== Navigational commands supported so far
* `WASD`: fly forward, left, right, and backward.
* `space/shift`: glide straight up or down
* `Mousewheel`: rotate 4D around the plane of your monitor
* `r/f`: strafe WINT and ZANT relative to your camera
* `t/g`: rotate 4D around the plane that intersects vertical and forward directions.
    This is functionally the same as looking 90 degrees left or right and then using mousewheel.
    
* `3`: toggle between up aligned and horizontal aligned view.
** You start out in `horizontal aligned` view. This works mostly like 4D miner, where Y is always the vertical axis but you can rotate the other 3 about in fun ways.
** toggling to `up aligned` view automatically aligns you to XZW so that the Y axis .. normally sacred vertical .. is now hidden.
    In this mode there is no sacred vertical direction, and mousewheel/t/g can really take you to some strange rotations!
    But exploring XZW in particular is quite informative.
** Toggling back to `horizontally aligned` forces Y to be vertical again.

== World file
The world file is currently a custom format not related to 4D Miner and limited to 32x32x32x32 in size.
roadmap includes bringing that into compatability with native 4D Miner world directories

== Textures
As of now this project includes a copy of tiles.png that came with 4D Miner.
If Mashpoe asks us to not distribute the file they created, then we'll probably set up a unique tile.png
which users can easily replace with the one from their official 4D Miner install.
It's UVW map is identical, and thus it can be edited by folk who want to toy with the textures.

== World Editing
Not supported yet, but on the roadmap.
