var canvas, ctx, flag = false,
        prevX = 0,
        currX = 0,
        prevY = 0,
        currY = 0,
        dot_flag = false;
        pixel_width = 10;
        translated_width = 20; // TRANSLATED_WIDTH = CANVAS_WIDTH / PIXEL_WIDTH


    var x = "black",
        y = 2;
    
    function init() {
        canvas = document.getElementById('can');
        ctx = canvas.getContext("2d", { willReadFrequently: true });
        w = canvas.width;
        h = canvas.height;
    
        canvas.addEventListener("mousemove", function (e) {
            findxy('move', e)
        }, false);
        canvas.addEventListener("mousedown", function (e) {
            findxy('down', e)
        }, false);
        canvas.addEventListener("mouseup", function (e) {
            findxy('up', e)
        }, false);
        canvas.addEventListener("mouseout", function (e) {
            findxy('out', e)
        }, false);
    }
/*
    function drawGrid() {
        ctx.strokeStyle = "white";
        for (let x = 0; x <= w; x += pixel_width) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }

        for (let y = 0; y <= h; y += pixel_width) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
    }
    */

    function draw() {
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
        ctx.lineTo(currX, currY);
        ctx.strokeStyle = x;
        ctx.lineWidth = y;
        ctx.stroke();
        ctx.closePath();
    }
    
    function erase() {
        ctx.clearRect(0, 0, w, h);
    }
    
    function findxy(res, e) {
        if (res == 'down') {
            prevX = currX;
            prevY = currY;
            currX = e.clientX - canvas.offsetLeft;
            currY = e.clientY - canvas.offsetTop;
    
            flag = true;
            dot_flag = true;
            if (dot_flag) {
                ctx.beginPath();
                ctx.fillStyle = x;
                ctx.fillRect(currX, currY, 2, 2);
                ctx.closePath();
                dot_flag = false;
            }
        }
        if (res == 'up' || res == "out") {
            flag = false;
        }
        if (res == 'move') {
            if (flag) {
                prevX = currX;
                prevY = currY;
                currX = e.clientX - canvas.offsetLeft;
                currY = e.clientY - canvas.offsetTop;
                draw();
            }
        }
    }

function getData(colorArray) {
    const imgData = ctx.getImageData(0, 0, w, h);
    
    // Initialize an empty array to store the color values.
    // Store the color values in the array.
    for (let i = 0; i < imgData.data.length; i += 4) {
        const r = imgData.data[i];     // Red
        const g = imgData.data[i+1];   // Green
        const b = imgData.data[i+2];   // Blue
        const a = imgData.data[i+3];   // Alpha
        colorArray.push({r, g, b, a});
    }

    // Log the array to the console.
    console.log(colorArray)
    return colorArray
}

function train() {
    const colorArray = []
    getData(colorArray)
    const trainArray = [];
    var digitVal = document.getElementById("digit").value;
    if (!digitVal || digitVal < 0) {
        alert("Please type and draw a digit value in order to train the network");
        return;
    }
    
    trainArray.push({"y0": colorArray, "label": parseInt(digitVal)});
    // Time to send a training batch to the server.
    alert("Sending training data to server...");
    var json = {
        trainArray: trainArray,
        train: true
    };

    sendData(json);
}

function recieveResposne(xmlHttp) {
    if (xmlHttp.status != 200) {
        alert("Server returned status " + xmlHttp.status);
        return;
    }
    var responseJSON = JSON.parse(xmlHttp.responseText);
    if (xmlHttp.responseText && responseJSON.type == "test") {
        alert("The neural network predicts you wrote a \'" 
               + responseJSON.result + '\'');
    }
}

function onError(e) {
    alert("Error occurred while connecting to server: " + e.target.statusText);
}

function test() {
    if (!colorArray) {
        alert("Please draw a digit in order to test the network");
        return;
    }
    var json = {
        image: colorArray,
        predict: true
    };
    sendData(json);
}

function sendData(json) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open('POST', this.HOST + ":" + this.PORT, false);
    xmlHttp.onload = function() { this.receiveResponse(xmlHttp); }.bind(this);
    xmlHttp.onerror = function() { this.onError(xmlHttp) }.bind(this);
    var msg = JSON.stringify(json);
    xmlHttp.setRequestHeader('Content-length', msg.length);
    xmlHttp.setRequestHeader("Connection", "close");
    xmlHttp.send(msg);
}