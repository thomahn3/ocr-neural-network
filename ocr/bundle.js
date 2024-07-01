(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
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
function erase() {
    ctx.clearRect(0, 0, w, h);
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

    saveData(json);
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
    saveData(json);
}

function saveData(json) {
    var msg = JSON.stringify(json);
    console.log(json)
    var fs = require('fs');
    fs.writeFile("data.txt", msg, function(err) {
        if (err) {
            console.log(err);
        }
    });
}
},{"fs":undefined}]},{},[1]);
