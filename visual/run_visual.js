var frameIndex = 0;
var data = null;
var frames = null;

var HABABO_LENGTH = 0;
var HABABO_WIDTH = 0;

function setup()
{
    can = createCanvas(700, 610);
    can.parent("sim");
    textAlign(CENTER);
    textSize(16);
    frameRate(6);
    noLoop();
}

function load_simulation()
{
    // convert json objects to frames
    frames = data["frames"];
    HABABO_LENGTH = data["HABABO_LENGTH"]
    HABABO_WIDTH = data["HABABO_WIDTH"]
    // technical lines
    document.getElementById("sim").style.display = ""; // show the simulation
    document.getElementById("load").style.display = "none"; // hide the load
    frameIndex = 0;
    loop();
}

function draw()
{
    if (frameIndex < frames.length)
    {
        background(200, 200, 200); // clear old frame
        display_hagabobim(frames[frameIndex]);
        strokeWeight(0);
        color(255);
        text("Frame #" + frameIndex, 10, 10);

        frameIndex++; // count this frame
    }
    else
    {
        document.getElementById("load").style.display = ""; // show the load - allow to load another one
        frameIndex = 0; // show in loop
    }
}

function display_hagabobim(hagabobim)
{
    for (var i = 0; i < hagabobim.length; i++)
    {
        noStroke();
        fill(255);
        push();
        translate(hagabobim[i][0], hagabobim[i][1]);
        rotate(hagabobim[i][2]);
        rect(HABABO_LENGTH/2, HABABO_WIDTH/2, HABABO_LENGTH, HABABO_WIDTH);
        pop();
    }
}

(function(){

    function onChange(event) {
        var reader = new FileReader();
        reader.onload = onReaderLoad;
        reader.readAsText(event.target.files[0]);
    }

    function onReaderLoad(event){
        console.log(event.target.result);
        data = JSON.parse(event.target.result); // get the data
        load_simulation(); // load the simulator
    }

    document.getElementById('simulation_file_upload').addEventListener('change', onChange);

}());