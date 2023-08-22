
(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('mapMapCtrl', mapMapCtrl);

    function mapMapCtrl($scope, $element) {
        var vm = this;
        var imgMovie = new Image();
        var imgMap = new Image();
        var imgBubblot = new Image();
        imgMovie.src = '../img/imgMovie.jpg';
        imgMap.src = '../img/fondMapDemo.jpg';
        imgBubblot.src = '../img/bubblotTrans.gif';
        var mapArea = document.getElementsByClassName('map-map');
        var fondMarin2 = document.getElementsByClassName('fond-marin2');
        //Add wheel and click event on the map for zooming and finding data
        for (var i = 0; i < mapArea.length; i++) {
            mapArea[i].addEventListener("wheel", computeZoom);
            //mapArea[i].addEventListener("mouseup", findData);
        }

        // Compute zoom when scrolling
        function computeZoom(element) {
            // find current location on screen 
            var clientRect = element.srcElement.getBoundingClientRect();
            var xScreen = event.screenX - clientRect.left;
            if (fondMarin2[3].style.display == "block") var yScreen = 2 * (event.screenY - clientRect.top);
            else var yScreen = 1 * (event.screenY - clientRect.top);

            // find current location on the image at the current scale
            $scope.ximage = $scope.ximage + ((xScreen - $scope.xlast) / $scope.zoom);
            $scope.yimage = $scope.yimage + ((yScreen - $scope.ylast) / $scope.zoom);

            // determine the new zoom scale
            $scope.zoom = $scope.zoom - event.deltaY / 300;

            // determine the location on the screen at the new scale
            $scope.xnew = (xScreen - $scope.ximage) / $scope.zoom;
            $scope.ynew = (yScreen - $scope.yimage) / $scope.zoom;

            // save the current screen location
            $scope.xlast = xScreen;
            $scope.ylast = yScreen;

            if ($scope.zoom <= 1) {
                $scope.zoom = 1;
                $scope.ximage = 0;
                $scope.yimage = 0;
                $scope.xnew = 0;
                $scope.ynew = 0;
                $scope.xlast = 0;
                $scope.ylast = 0;
            }
            $scope.$apply();
        }

        //Find data corresponding to the cursor position when clicking
        $scope.findData = function (event) {
            var clientRect = event.currentTarget.getBoundingClientRect();
            $scope.vaCursor = false;
            $scope.turbiCursor = false;
            $scope.magnCursor = false;
            $scope.movieCursor = false;
            $scope.infoMagnetism = [], $scope.infoTurbiRed = [], $scope.infoTurbiGreen = [], $scope.infoTurbiBlue = [], $scope.infoVa = [];
            //Convert cursor coordinates to map coordinates
            var infoX = $scope.ximage + ((event.screenX - clientRect.left - $scope.xlast) / $scope.zoom);
            if (fondMarin2[3].style.display == "block") var infoY = $scope.yimage + ((2 * (event.screenY - clientRect.top) - $scope.ylast) / $scope.zoom);
            else var infoY = $scope.yimage + ((1 * (event.screenY - clientRect.top) - $scope.ylast) / $scope.zoom);
            //Find if there is a data point at the cursor position
            //Loop for running through each bubblot 
            var dataFound = false;
            for (var j = 0; j < $scope.datax.length; j++) {
                //Loop for running through each coordinates
                for (var i = 0; i < $scope.datax[j].length; i++) {
                    //Comparing cursor position with coordinates of the data displayed
                    if (infoX < $scope.datax[j][i] + 10 / $scope.zoom && infoX > $scope.datax[j][i] - 10 / $scope.zoom &&
                        infoY < $scope.datay[j][i] + 10 / $scope.zoom && infoY > $scope.datay[j][i] - 10 / $scope.zoom &&
                        ($scope.displayVa && $scope.isVa[j][i] || $scope.displayTurbi && $scope.isTurbi[j][i] || $scope.displayMagn)) {
                        dataFound = true;
                        if ($scope.isVa[j][i] && $scope.displayVa) $scope.vaCursor = true;
                        else if ($scope.isTurbi[j][i] && $scope.displayTurbi) $scope.turbiCursor = true;
                        else if ($scope.isMovie[j][i] && $scope.displayMovie) $scope.movieCursor = true;
                        else if ($scope.displayMagn) $scope.magnCursor = true;

                        const viewUrl = "_design/by_date/_view/allData";
                        //Key to find the corresponding document in the database
                        const queryOptions = {
                            key: [$scope.dates[j][i][0], $scope.dates[j][i][1], $scope.dates[j][i][2], $scope.dates[j][i][3], $scope.dates[j][i][4], $scope.dates[j][i][5], j + 1]
                        };
                        couchBubblot1.get("bubblot", viewUrl, queryOptions).then(({ data, headers, status }) => {
                            //Extracting data corresponding to the cursor position
                            $scope.bubblotCursor = data.rows[0].key[6];
                            $scope.dateCursor = [data.rows[0].key[0], data.rows[0].key[1], data.rows[0].key[2], data.rows[0].key[3], data.rows[0].key[4], data.rows[0].key[5]];
                            $scope.latitudeCursor = parseInt($scope.dataxPump) + parseInt(data.rows[0].value.distancexToPump);
                            $scope.longitudeCursor = parseInt($scope.datayPump) + parseInt(data.rows[0].value.distanceyToPump);
                            $scope.depthCursor = data.rows[0].value.depth;
                            $scope.tempCursor = data.rows[0].value.temperature;
                            $scope.infoMagnetism = data.rows[0].value.magnetism.data;
                            $scope.infoTurbiRed = data.rows[0].value.turbidity.dataRed;
                            $scope.infoTurbiGreen = data.rows[0].value.turbidity.dataGreen;
                            $scope.infoTurbiBlue = data.rows[0].value.turbidity.dataBlue;
                            $scope.infoVa = data.rows[0].value.VA.data;
                            $scope.updatePanel = !$scope.updatePanel;
                            $scope.$apply()
                            // data is json response 
                            // headers is an object with all response headers 
                            // status is statusCode number 
                        }, err => {
                            console.log(err.code);
                            // either request error occured 
                            // ...or err.code=EDOCMISSING if document is missing 
                            // ...or err.code=EUNKNOWN if statusCode is unexpected 
                        });
                    }
                    if (dataFound == true) {
                        break;
                    }
                }
                if (dataFound == true) {
                    break;
                }
            }
            if (!dataFound){
                $scope.updatePanel = !$scope.updatePanel;
            }
        }
        $scope.playTime = -1;
        //Draw data
        $scope.drawData = function (element) {
            element.width = element.clientWidth;
            element.height = element.clientHeight;
            var context = element.getContext("2d");
            context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas
            context.globalAlpha = 1;
            context.lineWidth = 3 / $scope.zoom;
            context.translate($scope.ximage, $scope.yimage);
            context.scale($scope.zoom, $scope.zoom);
            context.translate($scope.xnew, $scope.ynew);
            context.translate(-$scope.ximage, -$scope.yimage);
            context.drawImage(imgMap, 0, 0, context.canvas.width, context.canvas.height);
            //Loop for running through each bubblot data
            for (var j = 0; j < $scope.datax.length; j++) {
                //Loop for running through each measurement point of bubblot j
                if ($scope.datax[j].length > 0) {
                    for (var i = 0; i < Math.min($scope.playTime + 1, $scope.datax[j].length); i++) {
                        //Corrdinates x y of the bubblot j
                        var x = $scope.datax[j][i];
                        var y = $scope.datay[j][i];
                        //Draw path of the bubblot j
                        context.lineWidth = 3 / $scope.zoom;
                        if (j == 0) context.strokeStyle = 'red';
                        else if (j == 1) context.strokeStyle = 'green';
                        else if (j == 2) context.strokeStyle = 'DodgerBlue';
                        if ($scope.displayPath && i > 0) {
                            //console.log($scope.dates[j][i], j, i)
                            //console.log($scope.dates[j][i-1], j, i-1)
                            var newDate = new Date($scope.dates[j][i][0], $scope.dates[j][i][1] - 1, $scope.dates[j][i][2], $scope.dates[j][i][3], $scope.dates[j][i][4], $scope.dates[j][i][5], 0);
                            var oldDate = new Date($scope.dates[j][i - 1][0], $scope.dates[j][i - 1][1] - 1, $scope.dates[j][i - 1][2], $scope.dates[j][i - 1][3], $scope.dates[j][i - 1][4], $scope.dates[j][i - 1][5], 0);
                            if (newDate - oldDate < 60000) {
                                context.beginPath();
                                context.moveTo($scope.datax[j][i - 1], $scope.datay[j][i - 1]);
                                context.lineTo(x, y);
                                context.closePath();
                                context.stroke();
                            }
                        }
                        //Draw magnetism of the bubblot j
                        if ($scope.displayMagn) {
                            context.strokeStyle = 'white';
                            context.lineWidth = 2 / $scope.zoom;
                            if ($scope.avgMagnetism[j][i] > 0.8) {
                                context.beginPath();
                                context.arc(x, y, 4 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                                context.beginPath();
                                context.arc(x, y, 8 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                                context.beginPath();
                                context.arc(x, y, 12 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                                context.beginPath();
                                context.arc(x, y, 16 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                            }
                            else if ($scope.avgMagnetism[j][i] > 0.6) {
                                context.beginPath();
                                context.arc(x, y, 4 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                                context.beginPath();
                                context.arc(x, y, 8 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                                context.beginPath();
                                context.arc(x, y, 12 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                            }
                            else if ($scope.avgMagnetism[j][i] > 0.4) {
                                context.beginPath();
                                context.arc(x, y, 4 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                                context.beginPath();
                                context.arc(x, y, 8 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                            }
                            else {
                                context.beginPath();
                                context.arc(x, y, 4 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                            }
                        }
                        //Draw VA for bubblot j
                        if ($scope.displayVa && $scope.isVa[j][i]) {
                            context.fillStyle = 'black';
                            context.beginPath();
                            context.arc(x, y, 9 / $scope.zoom, 0, 2 * Math.PI, false);
                            context.closePath();
                            context.fill();
                            context.strokeStyle = 'white';
                            context.lineWidth = 2 / $scope.zoom;
                            context.beginPath();
                            context.arc(x, y, 10 / $scope.zoom, 0, 2 * Math.PI, false);
                            context.closePath();
                            context.stroke();
                        }
                        //Draw Fe for bubblot j
                        if ($scope.displayFe && $scope.isFe[j][i]) {
                            context.fillStyle = 'red';
                            context.beginPath();
                            context.arc(x, y, 9 / $scope.zoom, 0, 2 * Math.PI, false);
                            context.closePath();
                            context.fill();
                        }
                        //Draw Pb for bubblot j
                        else if ($scope.displayPb && $scope.isPb[j][i]) {
                            context.fillStyle = '#798081';
                            context.beginPath();
                            context.arc(x, y, 9 / $scope.zoom, 0, 2 * Math.PI, false);
                            context.closePath();
                            context.fill();
                        }
                        //Draw Cu for bubblot j
                        else if ($scope.displayCu && $scope.isCu[j][i]) {
                            context.fillStyle = '#B36700';
                            context.beginPath();
                            context.arc(x, y, 9 / $scope.zoom, 0, 2 * Math.PI, false);
                            context.closePath();
                            context.fill();
                        }
                        //Draw Sn for bubblot j
                        else if ($scope.displaySn && $scope.isSn[j][i]) {
                            context.fillStyle = '#CECECE';
                            context.beginPath();
                            context.arc(x, y, 9 / $scope.zoom, 0, 2 * Math.PI, false);
                            context.closePath();
                            context.fill();
                        }
                        //Draw turbibity for bubblot j
                        if ($scope.displayTurbi && $scope.isTurbi[j][i]) {
                            var spiraleX = x, spiraleY = y;
                            var spiraleAngle = 0;
                            context.lineWidth = 3 / $scope.zoom;
                            //If no turbidity
                            if ($scope.avgTurbi[j][i][0] < 0.3 && $scope.avgTurbi[j][i][1] < 0.3 && $scope.avgTurbi[j][i][2] < 0.3) {
                                context.strokeStyle = 'aqua';
                                while (spiraleAngle < 6 * Math.PI) {
                                    context.beginPath();
                                    context.moveTo(spiraleX, spiraleY);
                                    spiraleX = x + 0.8 / $scope.zoom * spiraleAngle * Math.cos(spiraleAngle);
                                    spiraleY = y + 0.8 / $scope.zoom * spiraleAngle * Math.sin(spiraleAngle);
                                    context.lineTo(spiraleX, spiraleY);
                                    context.closePath();
                                    context.stroke();
                                    spiraleAngle = spiraleAngle + 2 * Math.PI / 60;
                                }
                            }
                            //If red turbidity
                            else if ($scope.avgTurbi[j][i][0] > $scope.avgTurbi[j][i][1] && $scope.avgTurbi[j][i][0] > $scope.avgTurbi[j][i][2]) {
                                context.strokeStyle = 'red';
                                while (spiraleAngle < $scope.avgTurbi[j][i][0] * 6 * Math.PI) {
                                    context.beginPath();
                                    context.moveTo(spiraleX, spiraleY);
                                    spiraleX = x + 0.8 / $scope.zoom * spiraleAngle * Math.cos(spiraleAngle);
                                    spiraleY = y + 0.8 / $scope.zoom * spiraleAngle * Math.sin(spiraleAngle);
                                    context.lineTo(spiraleX, spiraleY);
                                    context.closePath();
                                    context.stroke();
                                    spiraleAngle = spiraleAngle + 2 * Math.PI / 60;
                                }
                            }
                            //If green turbidity
                            else if ($scope.avgTurbi[j][i][1] > $scope.avgTurbi[j][i][2]) {
                                context.strokeStyle = 'green';
                                while (spiraleAngle < $scope.avgTurbi[j][i][1] * 6 * Math.PI) {
                                    context.beginPath();
                                    context.moveTo(spiraleX, spiraleY);
                                    spiraleX = x + 0.8 / $scope.zoom * spiraleAngle * Math.cos(spiraleAngle);
                                    spiraleY = y + 0.8 / $scope.zoom * spiraleAngle * Math.sin(spiraleAngle);
                                    context.lineTo(spiraleX, spiraleY);
                                    context.closePath();
                                    context.stroke();
                                    spiraleAngle = spiraleAngle + 2 * Math.PI / 60;
                                }
                            }
                            //If blue turbidity
                            else {
                                context.strokeStyle = 'dodgerblue';
                                while (spiraleAngle < $scope.avgTurbi[j][i][2] * 6 * Math.PI) {
                                    context.beginPath();
                                    context.moveTo(spiraleX, spiraleY);
                                    spiraleX = x + 0.8 / $scope.zoom * spiraleAngle * Math.cos(spiraleAngle);
                                    spiraleY = y + 0.8 / $scope.zoom * spiraleAngle * Math.sin(spiraleAngle);
                                    context.lineTo(spiraleX, spiraleY);
                                    context.closePath();
                                    context.stroke();
                                    spiraleAngle = spiraleAngle + 2 * Math.PI / 60;
                                }
                            }
                        }
                        //Draw movie for bubblot j
                        if ($scope.displayMovie && $scope.isMovie[j][i]) {
                            context.drawImage(imgMovie, x, y, 20 / $scope.zoom, 30 / $scope.zoom);
                        }
                    }
                    if ($scope.playTime >= $scope.datax[j].length - 1) context.drawImage(imgBubblot, $scope.datax[j][$scope.datax[j].length - 1] - 15 / $scope.zoom, $scope.datay[j][$scope.datax[j].length - 1] - 30 / $scope.zoom, 30 / $scope.zoom, 60 / $scope.zoom);
                    else context.drawImage(imgBubblot, $scope.datax[j][$scope.playTime] - 15 / $scope.zoom, $scope.datay[j][$scope.playTime] - 30 / $scope.zoom, 30 / $scope.zoom, 60 / $scope.zoom);
                }
            }
            //Draw pump
            if ($scope.displayPath) {
                context.fillStyle = 'orange';
                for (var i = 0; i < $scope.dataxPump.length; i++) {
                    //context.beginPath();
                    //context.arc($scope.dataxPump[i], $scope.datayPump[i], 8 / $scope.zoom, 0, 2 * Math.PI, false);
                    //context.closePath();
                    //context.fill();
                    context.drawImage(imgBubblot, $scope.dataxPump[i] - 15 / $scope.zoom, $scope.datayPump[i] - 30 / $scope.zoom, 30 / $scope.zoom, 60 / $scope.zoom);
                }
            }
            if ($scope.playData && $scope.playTime < Math.max($scope.datax[0].length, $scope.datax[1].length, $scope.datax[2].length) - 1) {
                setTimeout(function () { if ($scope.playData) $scope.playTime++; $scope.$apply()}, 100);
            }
            else if ($scope.playTime >= Math.max($scope.datax[0].length, $scope.datax[1].length, $scope.datax[2].length) - 1) $scope.playData = false;
        }
        //Draw the data 
        $scope.drawData2 = function (element) {
            element.width = element.clientWidth;
            element.height = element.clientHeight;
            var context = element.getContext("2d");
            context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas
            context.globalAlpha = 1;
            context.lineWidth = 3 / $scope.zoom;
            context.translate($scope.ximage, $scope.yimage);
            context.scale($scope.zoom, $scope.zoom);
            context.translate($scope.xnew, $scope.ynew);
            context.translate(-$scope.ximage, -$scope.yimage);
            context.drawImage(imgMap, 0, 0, context.canvas.width, context.canvas.height);

            //Loop for running through each bubblot data
            for (var j = 0; j < $scope.datax.length; j++) {
                //Loop for running through each measurement point of bubblot j
                if ($scope.datax[j]) {
                    for (var i = 0; i < $scope.datax[j].length; i++) {
                        //Corrdinates x y of the bubblot j
                        var x = $scope.datax[j][i];
                        var y = $scope.datay[j][i];
                        //Draw path of the bubblot j
                        context.lineWidth = 3 / $scope.zoom;
                        if (j == 0) context.strokeStyle = 'red';
                        else if (j == 1) context.strokeStyle = 'green';
                        else if (j == 2) context.strokeStyle = 'DodgerBlue';
                        if ($scope.displayPath && i < $scope.datax[j].length - 1) {
                            var newDate = new Date($scope.dates[j][i][0], $scope.dates[j][i][1] - 1, $scope.dates[j][i][2], $scope.dates[j][i][3], $scope.dates[j][i][4], $scope.dates[j][i][5], 0);
                            var oldDate = new Date($scope.dates[j][i + 1][0], $scope.dates[j][i + 1][1] - 1, $scope.dates[j][i + 1][2], $scope.dates[j][i + 1][3], $scope.dates[j][i + 1][4], $scope.dates[j][i + 1][5], 0);
                            if (oldDate - newDate < 60000) {
                                context.beginPath();
                                context.moveTo(x, y);
                                context.lineTo($scope.datax[j][i + 1], $scope.datay[j][i + 1]);
                                context.closePath();
                                context.stroke();
                            }
                        }
                        //Draw magnetism of the bubblot j
                        if ($scope.displayMagn) {
                            context.strokeStyle = 'white';
                            context.lineWidth = 2 / $scope.zoom;
                            if ($scope.avgMagnetism[j][i] > 0.8) {
                                context.beginPath();
                                context.arc(x, y, 4 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                                context.beginPath();
                                context.arc(x, y, 8 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                                context.beginPath();
                                context.arc(x, y, 12 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                                context.beginPath();
                                context.arc(x, y, 16 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                            }
                            else if ($scope.avgMagnetism[j][i] > 0.6) {
                                context.beginPath();
                                context.arc(x, y, 4 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                                context.beginPath();
                                context.arc(x, y, 8 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                                context.beginPath();
                                context.arc(x, y, 12 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                            }
                            else if ($scope.avgMagnetism[j][i] > 0.4) {
                                context.beginPath();
                                context.arc(x, y, 4 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                                context.beginPath();
                                context.arc(x, y, 8 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                            }
                            else {
                                context.beginPath();
                                context.arc(x, y, 4 / $scope.zoom, 0, 2 * Math.PI, false);
                                context.closePath();
                                context.stroke();
                            }
                        }
                        //Draw VA for bubblot j
                        if ($scope.displayVa && $scope.isVa[j][i]) {
                            context.fillStyle = 'black';
                            context.beginPath();
                            context.arc(x, y, 10 / $scope.zoom, 0, 2 * Math.PI, false);
                            context.closePath();
                            context.fill();
                            context.strokeStyle = 'white';
                            context.lineWidth = 2 / $scope.zoom;
                            context.beginPath();
                            context.arc(x, y, 10 / $scope.zoom, 0, 2 * Math.PI, false);
                            context.closePath();
                            context.stroke();
                        }
                        //Draw Fe for bubblot j
                        if ($scope.displayFe && $scope.isFe[j][i]) {
                            context.fillStyle = 'red';
                            context.beginPath();
                            context.arc(x, y, 10 / $scope.zoom, 0, 2 * Math.PI, false);
                            context.closePath();
                            context.fill();
                        }
                        //Draw Pb for bubblot j
                        else if ($scope.displayPb && $scope.isPb[j][i]) {
                            context.fillStyle = '#798081';
                            context.beginPath();
                            context.arc(x, y, 10 / $scope.zoom, 0, 2 * Math.PI, false);
                            context.closePath();
                            context.fill();
                        }
                        //Draw Cu for bubblot j
                        else if ($scope.displayCu && $scope.isCu[j][i]) {
                            context.fillStyle = '#B36700';
                            context.beginPath();
                            context.arc(x, y, 10 / $scope.zoom, 0, 2 * Math.PI, false);
                            context.closePath();
                            context.fill();
                        }
                        //Draw Sn for bubblot j
                        else if ($scope.displaySn && $scope.isSn[j][i]) {
                            context.fillStyle = '#CECECE';
                            context.beginPath();
                            context.arc(x, y, 10 / $scope.zoom, 0, 2 * Math.PI, false);
                            context.closePath();
                            context.fill();
                        }
                        //Draw turbibity for bubblot j
                        if ($scope.displayTurbi && $scope.isTurbi[j][i]) {
                            var spiraleX = x, spiraleY = y;
                            var spiraleAngle = 0;
                            context.lineWidth = 3 / $scope.zoom;
                            //If no turbidity
                            if ($scope.avgTurbi[j][i][0] < 0.3 && $scope.avgTurbi[j][i][1] < 0.3 && $scope.avgTurbi[j][i][2] < 0.3) {
                                context.strokeStyle = 'aqua';
                                while (spiraleAngle < 6 * Math.PI) {
                                    context.beginPath();
                                    context.moveTo(spiraleX, spiraleY);
                                    spiraleX = x + 0.8 / $scope.zoom * spiraleAngle * Math.cos(spiraleAngle);
                                    spiraleY = y + 0.8 / $scope.zoom * spiraleAngle * Math.sin(spiraleAngle);
                                    context.lineTo(spiraleX, spiraleY);
                                    context.closePath();
                                    context.stroke();
                                    spiraleAngle = spiraleAngle + 2 * Math.PI / 60;
                                }
                            }
                            //If red turbidity
                            else if ($scope.avgTurbi[j][i][0] > $scope.avgTurbi[j][i][1] && $scope.avgTurbi[j][i][0] > $scope.avgTurbi[j][i][2]) {
                                context.strokeStyle = 'red';
                                while (spiraleAngle < $scope.avgTurbi[j][i][0] * 6 * Math.PI) {
                                    context.beginPath();
                                    context.moveTo(spiraleX, spiraleY);
                                    spiraleX = x + 0.8 / $scope.zoom * spiraleAngle * Math.cos(spiraleAngle);
                                    spiraleY = y + 0.8 / $scope.zoom * spiraleAngle * Math.sin(spiraleAngle);
                                    context.lineTo(spiraleX, spiraleY);
                                    context.closePath();
                                    context.stroke();
                                    spiraleAngle = spiraleAngle + 2 * Math.PI / 60;
                                }
                            }
                            //If green turbidity
                            else if ($scope.avgTurbi[j][i][1] > $scope.avgTurbi[j][i][2]) {
                                context.strokeStyle = 'green';
                                while (spiraleAngle < $scope.avgTurbi[j][i][1] * 6 * Math.PI) {
                                    context.beginPath();
                                    context.moveTo(spiraleX, spiraleY);
                                    spiraleX = x + 0.8 / $scope.zoom * spiraleAngle * Math.cos(spiraleAngle);
                                    spiraleY = y + 0.8 / $scope.zoom * spiraleAngle * Math.sin(spiraleAngle);
                                    context.lineTo(spiraleX, spiraleY);
                                    context.closePath();
                                    context.stroke();
                                    spiraleAngle = spiraleAngle + 2 * Math.PI / 60;
                                }
                            }
                            //If blue turbidity
                            else {
                                context.strokeStyle = 'dodgerblue';
                                while (spiraleAngle < $scope.avgTurbi[j][i][2] * 6 * Math.PI) {
                                    context.beginPath();
                                    context.moveTo(spiraleX, spiraleY);
                                    spiraleX = x + 0.8 / $scope.zoom * spiraleAngle * Math.cos(spiraleAngle);
                                    spiraleY = y + 0.8 / $scope.zoom * spiraleAngle * Math.sin(spiraleAngle);
                                    context.lineTo(spiraleX, spiraleY);
                                    context.closePath();
                                    context.stroke();
                                    spiraleAngle = spiraleAngle + 2 * Math.PI / 60;
                                }
                            }
                        }
                        //Draw movie for bubblot j
                        if ($scope.displayMovie && $scope.isMovie[j][i]) {
                            context.drawImage(imgMovie, x, y, 20 / $scope.zoom, 30 / $scope.zoom);
                        }
                    }
                    if (i == $scope.datax[j].length) {
                        context.drawImage(imgBubblot, x - 15 / $scope.zoom, y - 30 / $scope.zoom, 30 / $scope.zoom, 60 / $scope.zoom);
                    }
                }
            }
            //Draw pump
            if ($scope.displayPath) {
                context.fillStyle = 'orange';
                for (var i = 0; i < $scope.dataxPump.length; i++) {
                    //context.beginPath();
                    //context.arc($scope.dataxPump[i], $scope.datayPump[i], 8 / $scope.zoom, 0, 2 * Math.PI, false);
                    //context.closePath();
                    //context.fill();
                    context.drawImage(imgBubblot, $scope.dataxPump[i] - 15 / $scope.zoom, $scope.datayPump[i] - 30 / $scope.zoom, 30 / $scope.zoom, 60 / $scope.zoom);
                }
            }

        }
    }

}());
