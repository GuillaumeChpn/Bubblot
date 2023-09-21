(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('mapPanelCtrl', mapPanelCtrl);

    function mapPanelCtrl($scope, $element) {
        var vm = this;
        var element = $element.find(".caleandar")[0];
        $scope.graphName = "V-A";
        //Initialiazing variables
        $scope.datax[0] = [], $scope.datay[0] = [], $scope.datax[1] = [], $scope.datay[1] = [], $scope.datax[2] = [], $scope.datay[2] = [];
        $scope.avgMagnetism[0] = [], $scope.avgMagnetism[1] = [], $scope.avgMagnetism[2] = [];
        $scope.isVa[0] = [], $scope.isVa[1] = [], $scope.isVa[2] = [];
        $scope.isFe[0] = [], $scope.isFe[1] = [], $scope.isFe[2] = [];
        $scope.isPb[0] = [], $scope.isPb[1] = [], $scope.isPb[2] = [];
        $scope.isCu[0] = [], $scope.isCu[1] = [], $scope.isCu[2] = [];
        $scope.isSn[0] = [], $scope.isSn[1] = [], $scope.isSn[2] = [];
        $scope.isTurbi[0] = [], $scope.isTurbi[1] = [], $scope.isTurbi[2] = [];
        $scope.avgTurbi[0] = [], $scope.avgTurbi[1] = [], $scope.avgTurbi[2] = [];
        $scope.isMovie[0] = [], $scope.isMovie[1] = [], $scope.isMovie[2] = [];
        $scope.dates[0] = [], $scope.dates[1] = [], $scope.dates[2] = [];

        var btnMagnElement = $element.find("canvas")[1];
        var contextMagn = btnMagnElement.getContext("2d");
        contextMagn.scale(1, 0.5);
        contextMagn.strokeStyle = 'white';
        contextMagn.globalAlpha = 1;
        contextMagn.lineWidth = 12;
        contextMagn.beginPath();
        contextMagn.arc(btnMagnElement.width / 2, btnMagnElement.height, 20, 0, 2 * Math.PI, false);
        contextMagn.closePath();
        contextMagn.stroke();
        contextMagn.beginPath();
        contextMagn.arc(btnMagnElement.width / 2, btnMagnElement.height, 50, 0, 2 * Math.PI, false);
        contextMagn.closePath();
        contextMagn.stroke();
        contextMagn.beginPath();
        contextMagn.arc(btnMagnElement.width / 2, btnMagnElement.height, 80, 0, 2 * Math.PI, false);
        contextMagn.closePath();
        contextMagn.stroke();
        contextMagn.beginPath();
        contextMagn.arc(btnMagnElement.width / 2, btnMagnElement.height, 110, 0, 2 * Math.PI, false);
        contextMagn.closePath();
        contextMagn.stroke();

        var btnTurbiElement = $element.find("canvas")[3]
        var contextTurbi = btnTurbiElement.getContext("2d");
        contextTurbi.scale(1, 0.5);
        contextTurbi.strokeStyle = 'aqua';
        contextTurbi.lineWidth = 12;
        var x = btnTurbiElement.width / 2
        var y = btnTurbiElement.height
        var spiraleX = x, spiraleY = y;
        var spiraleAngle = 0;
        while (spiraleAngle < 6 * Math.PI) {
            contextTurbi.beginPath();
            contextTurbi.moveTo(spiraleX, spiraleY);
            spiraleX = x + 6 / $scope.zoom * spiraleAngle * Math.cos(spiraleAngle);
            spiraleY = y + 6 / $scope.zoom * spiraleAngle * Math.sin(spiraleAngle);
            contextTurbi.lineTo(spiraleX, spiraleY);
            contextTurbi.closePath();
            contextTurbi.stroke();
            spiraleAngle = spiraleAngle + 2 * Math.PI / 60;
        }

        var btnPathElement = $element.find("canvas")[0]
        var contextPath = btnPathElement.getContext("2d");
        contextPath.scale(1, 0.5);
        var imgLocation = new Image();
        imgLocation.src = '../img/locationIcon.png';
        imgLocation.onload = function () {
            contextPath.drawImage(imgLocation, btnPathElement.width / 12, btnPathElement.height / 6, btnPathElement.width / 1.2, btnPathElement.height / 0.6);
        };

        var btnVideoElement = $element.find("canvas")[2]
        var contextVideo = btnVideoElement.getContext("2d");
        contextVideo.scale(1, 0.5);
        var imgVideo = new Image();
        imgVideo.src = '../img/videoIcon.png';
        imgVideo.onload = function () {
            contextVideo.drawImage(imgVideo, btnVideoElement.width / 12, btnVideoElement.height / 6, btnVideoElement.width / 1.2, btnVideoElement.height / 0.6);
        };

        var btnVaElement = $element.find("canvas")[4]
        var contextVa = btnVaElement.getContext("2d");
        contextVa.scale(1, 0.5);
        var imgVa = new Image();
        imgVa.src = '../img/vaIcon.png';
        imgVa.onload = function () {
            contextVa.drawImage(imgVa, btnVaElement.width / 12, btnVaElement.height / 6, btnVaElement.width / 1.2, btnVaElement.height / 0.6);
        };


        //Callback functions for date input to update fond-marin and fond-marin2
        var startDay = document.getElementsByClassName("start-day");
        var startTime = document.getElementsByClassName("start-time");
        var stopDay = document.getElementsByClassName("stop-day");
        var stopTime = document.getElementsByClassName("stop-time");
        var submit = document.getElementsByClassName("submit-time");
        for (var i = 0; i < submit.length; i++) {
            if (i == 0) {
                startDay[i].addEventListener('change', updateDateDown);
                startTime[i].addEventListener('change', updateDateDown);
                stopDay[i].addEventListener('change', updateDateDown);
                stopTime[i].addEventListener('change', updateDateDown);
            }
            else {
                startDay[i].addEventListener('change', updateDateUp);
                startTime[i].addEventListener('change', updateDateUp);
                stopDay[i].addEventListener('change', updateDateUp);
                stopTime[i].addEventListener('change', updateDateUp);
            }
        }
        function updateDateDown() {
            startDay[1].value = startDay[0].value;
            startTime[1].value = startTime[0].value;
            stopDay[1].value = stopDay[0].value;
            stopTime[1].value = stopTime[0].value;
        }
        function updateDateUp() {
            startDay[0].value = startDay[1].value;
            startTime[0].value = startTime[1].value;
            stopDay[0].value = stopDay[1].value;
            stopTime[0].value = stopTime[1].value;
        }
        $scope.playDataPressed = function () {
            $scope.playData = !$scope.playData;
        }
        //Extract data from the database into the program memory when clicking submit
        $scope.extractData = function (element) {
            //Get time slot 
            var dataStartYear = parseInt(startDay[0].value[0] + startDay[0].value[1] + startDay[0].value[2] + startDay[0].value[3]);
            var dataStartMonth = parseInt(startDay[0].value[5] + startDay[0].value[6]);
            var dataStartDay = parseInt(startDay[0].value[8] + startDay[0].value[9]);

            var dataEndYear = parseInt(stopDay[0].value[0] + stopDay[0].value[1] + stopDay[0].value[2] + stopDay[0].value[3]);
            var dataEndMonth = parseInt(stopDay[0].value[5] + stopDay[0].value[6]);
            var dataEndDay = parseInt(stopDay[0].value[8] + stopDay[0].value[9]);

            //Initialiazing data
            $scope.dataxPump = [], $scope.datayPump = [], $scope.datax[0] = [], $scope.datay[0] = [], $scope.datax[1] = [], $scope.datay[1] = [], $scope.datax[2] = [], $scope.datay[2] = [];
            $scope.avgMagnetism[0] = [], $scope.avgMagnetism[1] = [], $scope.avgMagnetism[2] = [];
            $scope.isVa[0] = [], $scope.isVa[1] = [], $scope.isVa[2] = [];
            $scope.isFe[0] = [], $scope.isFe[1] = [], $scope.isFe[2] = [];
            $scope.isPb[0] = [], $scope.isPb[1] = [], $scope.isPb[2] = [];
            $scope.isCu[0] = [], $scope.isCu[1] = [], $scope.isCu[2] = [];
            $scope.isSn[0] = [], $scope.isSn[1] = [], $scope.isSn[2] = [];
            $scope.isTurbi[0] = [], $scope.isTurbi[1] = [], $scope.isTurbi[2] = [];
            $scope.avgTurbi[0] = [], $scope.avgTurbi[1] = [], $scope.avgTurbi[2] = [];
            $scope.isMovie[0] = [], $scope.isMovie[1] = [], $scope.isMovie[2] = [];
            $scope.dates[0] = [], $scope.dates[1] = [], $scope.dates[2] = [];

            const viewUrl = "_design/by_date/_view/allData";

            //Database key
            const queryOptions = {
                startkey: [dataStartYear, dataStartMonth, dataStartDay, 0, 0, 0, 1],
                endkey: [dataEndYear, dataEndMonth, dataEndDay, 0, 0, 0, 3]
            };
            //Searching in the database
            couchBubblot1.get("bubblot", viewUrl, queryOptions).then(({ data, headers, status }) => {
                //Loop for running through each data received
                for (var i = 0; i < data.rows.length; i++) {
                    //Get from which bubblot comes the data
                    var bubblotIndex = data.rows[i].key[6] - 1;
                    //Get the date of the measurement
                    $scope.dates[bubblotIndex].push([data.rows[i].key[0], data.rows[i].key[1], data.rows[i].key[2], data.rows[i].key[3], data.rows[i].key[4], data.rows[i].key[5]]);
                    if (i == 0) {
                        $scope.dataxPump.push(500);
                        $scope.datayPump.push(600);
                    }
                    //Get x y coordinates
                    $scope.datax[bubblotIndex].push(parseInt($scope.dataxPump) + parseInt(data.rows[i].value.distancexToPump));
                    $scope.datay[bubblotIndex].push(parseInt($scope.datayPump) + parseInt(data.rows[i].value.distanceyToPump));
                    //Get magnetism
                    $scope.avgMagnetism[bubblotIndex].push(parseFloat(data.rows[i].value.magnetism.mean));
                    //Get if there is VA
                    if (data.rows[i].value.VA.available) $scope.isVa[bubblotIndex].push(true);
                    else $scope.isVa[bubblotIndex].push(false);
                    //Get if there is Fe
                    if (data.rows[i].value.VA.fe) $scope.isFe[bubblotIndex].push(true);
                    else $scope.isFe[bubblotIndex].push(false);
                    //Get if there is Pb
                    if (data.rows[i].value.VA.pb) $scope.isPb[bubblotIndex].push(true);
                    else $scope.isPb[bubblotIndex].push(false);
                    //Get if there is Cu
                    if (data.rows[i].value.VA.cu) $scope.isCu[bubblotIndex].push(true);
                    else $scope.isCu[bubblotIndex].push(false);
                    //Get if there is Sn
                    if (data.rows[i].value.VA.sn) $scope.isSn[bubblotIndex].push(true);
                    else $scope.isSn[bubblotIndex].push(false);
                    //Get if there is turbidity
                    if (data.rows[i].value.turbidity.available) {
                        $scope.isTurbi[bubblotIndex].push(true);
                        //Get turbidity average values for each color
                        $scope.avgTurbi[bubblotIndex].push([data.rows[i].value.turbidity.avgRed, data.rows[i].value.turbidity.avgGreen, data.rows[i].value.turbidity.avgBlue]);
                    }
                    else {
                        $scope.isTurbi[bubblotIndex].push(false);
                        $scope.avgTurbi[bubblotIndex].push([null, null, null]);
                    }
                    //Get if there is a movie
                    if (data.rows[i].value.movie.available) $scope.isMovie[bubblotIndex].push(true);
                    else $scope.isMovie[bubblotIndex].push(false);
                }
                $scope.extractingData = !$scope.extractingData;
                $scope.$apply();
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
        //Init datasets for graph
        var initData = {
            datasets: [{
                borderColor: 'white',
                pointBackgroundColor: 'white',
                borderWidth: 2,
                pointRadius: 1,
                data: 0,
            }]
        };
        //Init graph in information panel 
        var graphInfoData;
        var context = $element.find("canvas")[5].getContext("2d");
        var graphInfo = new Chart(context, {
            type: 'line',
            data: initData,
            options: {
                maintainAspectRatio: false,
                legend: {
                    display: false,
                },
                tooltips: {
                    enabled: false,
                },
                scales: {
                    xAxes: [{
                        type: 'linear',
                        position: 'bottom',
                        scaleLabel: {
                            display: true,
                            fontColor: "white",
                            fontSize: 20,
                            labelString: 'V',
                        },
                        ticks: {
                            min: 0,
                            max: 5,
                            stepSize: 1,
                            fontColor: "white"
                        },
                        gridLines: {
                            display: true,
                            color: "white",
                        }
                    }],
                    yAxes: [{
                        type: 'linear',
                        position: 'left',
                        scaleLabel: {
                            display: true,
                            fontColor: "white",
                            fontSize: 20,
                            labelString: 'mA',
                        },
                        ticks: {
                            min: 0,
                            max: 1,
                            stepSize: 0.2,
                            fontColor: "white"
                        },
                        gridLines: {
                            display: true,
                            color: "white",
                        }
                    }]
                }
            }
        });
        //Update the information graph when clicking on map
        $scope.updateInfoPanel = function (element) {
            //If cursor is on VA spot, plot V-A graph
            if ($scope.vaCursor) {
                while (graphInfo.data.datasets.length > 0) {
                    graphInfo.data.datasets.pop();
                };
                graphInfo.data.datasets.push({
                    borderColor: 'white',
                    pointBackgroundColor: 'white',
                    borderWidth: 2,
                    pointRadius: 1,
                    data: $scope.infoVa,
                });
                graphInfo.options.scales.xAxes[0].ticks.max = 30;
                graphInfo.options.scales.xAxes[0].ticks.stepSize = 5;
                graphInfo.options.scales.xAxes[0].scaleLabel.labelString = "V";
                graphInfo.options.scales.yAxes[0].ticks.max = 20;
                graphInfo.options.scales.yAxes[0].ticks.stepSize = 5;
                graphInfo.options.scales.yAxes[0].scaleLabel.labelString = "mA";
                $scope.graphName = "V-A";
            }
            //If cursor click on turbidity spot, plot turbi graph
            else if ($scope.turbiCursor) {
                while (graphInfo.data.datasets.length > 0) {
                    graphInfo.data.datasets.pop();
                };
                graphInfo.data.datasets.push({
                    borderColor: 'red',
                    pointBackgroundColor: 'red',
                    borderWidth: 2,
                    pointRadius: 1,
                    data: $scope.infoTurbiRed,
                });
                graphInfo.data.datasets.push({
                    borderColor: 'green',
                    pointBackgroundColor: 'green',
                    borderWidth: 2,
                    pointRadius: 1,
                    data: $scope.infoTurbiGreen,
                });
                graphInfo.data.datasets.push({
                    borderColor: 'DodgerBlue',
                    pointBackgroundColor: 'DodgerBlue',
                    borderWidth: 2,
                    pointRadius: 1,
                    data: $scope.infoTurbiBlue,
                });
                graphInfo.options.scales.xAxes[0].ticks.max = 30;
                graphInfo.options.scales.xAxes[0].ticks.stepSize = 5;
                graphInfo.options.scales.xAxes[0].scaleLabel.labelString = "s";
                graphInfo.options.scales.yAxes[0].ticks.max = 100;
                graphInfo.options.scales.yAxes[0].ticks.stepSize = 20;
                graphInfo.options.scales.yAxes[0].scaleLabel.labelString = "%";
                $scope.graphName = "Turbidity";
            }
            //If cursor click on magnetic spot, plot magn graph
            else if ($scope.magnCursor) {
                while (graphInfo.data.datasets.length > 0) {
                    graphInfo.data.datasets.pop();
                };
                graphInfo.data.datasets.push({
                    borderColor: 'white',
                    pointBackgroundColor: 'white',
                    borderWidth: 2,
                    pointRadius: 1,
                    data: $scope.infoMagnetism,
                });
                graphInfo.data.datasets[0].data = $scope.infoMagnetism;
                graphInfo.options.scales.xAxes[0].ticks.max = 30;
                graphInfo.options.scales.xAxes[0].ticks.stepSize = 5;
                graphInfo.options.scales.xAxes[0].scaleLabel.labelString = "s";
                graphInfo.options.scales.yAxes[0].ticks.max = 5;
                graphInfo.options.scales.yAxes[0].ticks.stepSize = 1;
                graphInfo.options.scales.yAxes[0].scaleLabel.labelString = "T";
                $scope.graphName = "Magnetism";
            }
            //If cursor click on movie spot, plot movie
            else if ($scope.movieCursor) {
                $scope.graphName = "Video";
            }
            graphInfo.update();
        };
        function drawGraphVa(context) {
            var dataBar = [], dataX = [];

            for (var i = 0; i < 30; i++) {
                dataX.push("");
                if (i == 7 || i == 12 || i == 19 || i == 22) {
                    dataBar.push(1);
                }
                else {
                    dataBar.push(0);
                }
            }
            var vaData = {
                labels: dataX,
                datasets: [{
                    type: 'bar',
                    data: dataBar,
                    fill: true,
                    backgroundColor: 'red',
                    borderColor: 'red',
                },
                {
                    type: 'line',
                    data: $scope.infoVa,
                    borderColor: 'white',
                    pointBackgroundColor: 'white',
                    borderWidth: 2,
                    pointRadius: 1,
                }]
            };
            graphInfo = new Chart(context, {
                type: "bar",
                data: vaData,
                options: {
                    maintainAspectRatio: false,
                    legend: {
                        display: false,
                    },
                    tooltips: {
                        enabled: false,
                    },
                    scales: {
                        xAxes: [{
                            position: "bottom",
                            display: true,
                            barPercentage: 0.8,
                            gridLines: {
                                display: false,
                                color: "white",
                            },
                            labels: {
                                show: false,
                            },
                            scaleLabel: {
                                display: true,
                                fontColor: "white",
                                fontSize: 20,
                                labelString: '[V]',
                            }
                        }],
                        yAxes: [{
                            type: 'linear',
                            position: 'left',
                            scaleLabel: {
                                display: true,
                                fontColor: "white",
                                fontSize: 20,
                                labelString: '[mA]',
                            },
                            ticks: {
                                min: 0,
                                max: 1,
                                fontColor: "white"
                            },
                            gridLines: {
                                display: true,
                                color: "white",
                            }
                        }]
                    }
                }
            });
        }
        function drawGraphMagnetism(context) {
            var magnetismData = {
                datasets: [{
                    borderColor: 'white',
                    pointBackgroundColor: 'white',
                    pointHoverBackgroundColor: '#000',
                    pointHoverBorderColor: '#000',
                    borderWidth: 2,
                    pointRadius: 1,
                    data: $scope.infoMagnetism
                }]
            };
            graphInfo = new Chart(context, {
                type: 'line',
                data: magnetismData,
                options: {
                    maintainAspectRatio: false,
                    legend: {
                        display: false,
                    },
                    tooltips: {
                        enabled: false,
                    },
                    scales: {
                        xAxes: [{
                            type: 'linear',
                            position: 'bottom',
                            scaleLabel: {
                                display: true,
                                fontColor: "white",
                                fontSize: 20,
                                labelString: '[s]',
                            },
                            ticks: {
                                min: 0,
                                max: 30,
                                stepSize: 5,
                                fontColor: "white"
                            },
                            gridLines: {
                                display: true,
                                color: "white",
                            }
                        }],
                        yAxes: [{
                            type: 'linear',
                            position: 'left',
                            scaleLabel: {
                                display: true,
                                fontColor: "white",
                                fontSize: 20,
                                labelString: '[T]',
                            },
                            ticks: {
                                min: 0,
                                max: 5,
                                fontColor: "white"
                            },
                            gridLines: {
                                display: true,
                                color: "white",
                            }
                        }]
                    }
                }
            });
        }
        function drawGraphTurbi(context) {
            var turbiData = {
                datasets: [{
                    borderColor: 'red',
                    pointBackgroundColor: 'red',
                    pointHoverBackgroundColor: '#000',
                    pointHoverBorderColor: '#000',
                    borderWidth: 2,
                    pointRadius: 1,
                    data: $scope.infoTurbiRed
                }, {
                    borderColor: 'green',
                    pointBackgroundColor: 'green',
                    pointHoverBackgroundColor: '#000',
                    pointHoverBorderColor: '#000',
                    borderWidth: 2,
                    pointRadius: 1,
                    data: $scope.infoTurbiGreen
                }, {
                    borderColor: 'blue',
                    pointBackgroundColor: 'blue',
                    pointHoverBackgroundColor: '#000',
                    pointHoverBorderColor: '#000',
                    borderWidth: 2,
                    pointRadius: 1,
                    data: $scope.infoTurbiBlue
                }
                ]
            };
            graphInfo = new Chart(context, {
                type: 'line',
                data: turbiData,
                options: {
                    maintainAspectRatio: false,
                    legend: {
                        display: false,
                    },
                    tooltips: {
                        enabled: false,
                    },
                    scales: {
                        xAxes: [{
                            type: 'linear',
                            position: 'bottom',
                            scaleLabel: {
                                display: true,
                                fontColor: "white",
                                fontSize: 20,
                                labelString: '[s]',
                            },
                            ticks: {
                                min: 0,
                                max: 30,
                                stepSize: 5,
                                fontColor: "white"
                            },
                            gridLines: {
                                display: true,
                                color: "white",
                            }
                        }],
                        yAxes: [{
                            type: 'linear',
                            position: 'left',
                            scaleLabel: {
                                display: true,
                                fontColor: "white",
                                fontSize: 20,
                                labelString: '[%]',
                            },
                            ticks: {
                                min: 0,
                                max: 100,
                                stepSize: 20,
                                fontColor: "white"
                            },
                            gridLines: {
                                display: true,
                                color: "white",
                            }
                        }]
                    }
                }
            });
        }
        /*
  Author: Jack Ducasse;
  Version: 0.1.0;
  (◠‿◠✿)
*/      $scope.updateCalendarRwd = function () {
            if(objCalendar) createCalendar(element, -1);
        }
        $scope.updateCalendarFwd = function () {
            if(objCalendar) createCalendar(element, 1);
        }
        function createCalendar(element, adjuster) {
            if (typeof adjuster !== 'undefined') {
                var newDate = new Date(objCalendar.Selected.Year, objCalendar.Selected.Month + adjuster, 1);
                objCalendar = new Calendar(objCalendar.Model, objCalendar.Options, newDate);
                element.innerHTML = '';
            } else {
                for (var key in objCalendar.Options) {
                    typeof objCalendar.Options[key] != 'function' && typeof objCalendar.Options[key] != 'object' && objCalendar.Options[key] ? element.className += " " + key + "-" + objCalendar.Options[key] : 0;
                }
            }
            var months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

            function AddSidebar() {
                var sidebar = document.createElement('div');
                sidebar.className += 'cld-sidebar';

                var monthList = document.createElement('ul');
                monthList.className += 'cld-monthList';

                for (var i = 0; i < months.length - 3; i++) {
                    var x = document.createElement('li');
                    x.className += 'cld-month';
                    var n = i - (4 - objCalendar.Selected.Month);
                    // Account for overflowing month values
                    if (n < 0) { n += 12; }
                    else if (n > 11) { n -= 12; }
                    // Add Appropriate Class
                    if (i == 0) {
                        x.className += ' cld-rwd cld-nav';
                        x.addEventListener('click', function () {
                            typeof objCalendar.Options.ModelChange == 'function' ? objCalendar.Model = objCalendar.Options.ModelChange() : objCalendar.Model = objCalendar.Options.ModelChange;
                            createCalendar(element, -1);

                        });
                        x.innerHTML += '<svg height="15" width="15" viewBox="0 0 100 75" fill="rgba(255,255,255,0.5)"><polyline points="0,75 100,75 50,0"></polyline></svg>';
                    }
                    else if (i == months.length - 4) {
                        x.className += ' cld-fwd cld-nav';
                        x.addEventListener('click', function () {
                            typeof objCalendar.Options.ModelChange == 'function' ? objCalendar.Model = objCalendar.Options.ModelChange() : objCalendar.Model = objCalendar.Options.ModelChange;
                            createCalendar(element, 1);
                        });
                        x.innerHTML += '<svg height="15" width="15" viewBox="0 0 100 75" fill="rgba(255,255,255,0.5)"><polyline points="0,0 100,0 50,75"></polyline></svg>';
                    }
                    else {
                        if (i < 4) { x.className += ' cld-pre'; }
                        else if (i > 4) { x.className += ' cld-post'; }
                        else { x.className += ' cld-curr'; }

                        //prevent losing var adj value (for whatever reason that is happening)
                        (function () {
                            var adj = (i - 4);
                            //x.addEventListener('click', function(){createCalendar(calendar, element, adj);console.log('kk', adj);} );
                            x.addEventListener('click', function () {
                                typeof objCalendar.Options.ModelChange == 'function' ? objCalendar.Model = objCalendar.Options.ModelChange() : objCalendar.Model = objCalendar.Options.ModelChange;
                                createCalendar(element, adj);
                            });
                            x.setAttribute('style', 'opacity:' + (1 - Math.abs(adj) / 4));
                            x.innerHTML += months[n].substr(0, 3);
                        }()); // immediate invocation

                        if (n == 0) {
                            var y = document.createElement('li');
                            y.className += 'cld-year';
                            if (i < 5) {
                                y.innerHTML += objCalendar.Selected.Year;
                            } else {
                                y.innerHTML += objCalendar.Selected.Year + 1;
                            }
                            monthList.appendChild(y);
                        }
                    }
                    monthList.appendChild(x);
                }
                sidebar.appendChild(monthList);
                if (objCalendar.Options.NavLocation) {
                    document.getElementById(objCalendar.Options.NavLocation).innerHTML = "";
                    document.getElementById(objCalendar.Options.NavLocation).appendChild(sidebar);
                }
                else { element.appendChild(sidebar); }
            }

            var mainSection = document.createElement('div');
            mainSection.className += "cld-main";

            function AddDateTime() {
                var datetime = document.createElement('div');
                datetime.className += "cld-datetime";
                if (objCalendar.Options.NavShow && !objCalendar.Options.NavVertical) {
                    var rwd = document.createElement('div');
                    rwd.className += " cld-rwd cld-nav";
                    rwd.addEventListener('click', function () {$scope.calendarRwd=!$scope.calendarRwd;$scope.$apply();});
                    rwd.innerHTML = '<svg height="15" width="15" viewBox="0 0 75 100" fill="rgba(0,0,0,0.5)"><polyline points="0,50 75,0 75,100"></polyline></svg>';
                    datetime.appendChild(rwd);
                }
                var today = document.createElement('div');
                today.className += ' today';
                today.innerHTML = months[objCalendar.Selected.Month] + ", " + objCalendar.Selected.Year;
                datetime.appendChild(today);
                if (objCalendar.Options.NavShow && !objCalendar.Options.NavVertical) {
                    var fwd = document.createElement('div');
                    fwd.className += " cld-fwd cld-nav";
                    fwd.addEventListener('click', function () {$scope.calendarFwd=!$scope.calendarFwd;$scope.$apply(); });
                    fwd.innerHTML = '<svg height="15" width="15" viewBox="0 0 75 100" fill="rgba(0,0,0,0.5)"><polyline points="0,0 75,50 0,100"></polyline></svg>';
                    datetime.appendChild(fwd);
                }
                if (objCalendar.Options.DatetimeLocation) {
                    document.getElementById(objCalendar.Options.DatetimeLocation).innerHTML = "";
                    document.getElementById(objCalendar.Options.DatetimeLocation).appendChild(datetime);
                }
                else { mainSection.appendChild(datetime); }
            }

            function AddLabels() {
                var labels = document.createElement('ul');
                labels.className = 'cld-labels';
                var labelsList = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
                for (var i = 0; i < labelsList.length; i++) {
                    var label = document.createElement('li');
                    label.className += "cld-label";
                    label.innerHTML = labelsList[i];
                    labels.appendChild(label);
                }
                mainSection.appendChild(labels);
            }
            function AddDays() {
                // Create Number Element    
                function DayNumber(n) {
                    var number = document.createElement('p');
                    number.className += "cld-number";
                    number.innerHTML += n;
                    number.addEventListener("mouseup", function () {
                        //Get time slot 
                        var dataStartYear = objCalendar.Selected.Year;
                        var dataStartMonth = objCalendar.Selected.Month+1;
                        var dataStartDay = parseInt(number.innerHTML);

                        var dataEndYear = objCalendar.Selected.Year;
                        var dataEndMonth = objCalendar.Selected.Month+1;
                        var dataEndDay = parseInt(number.innerHTML);

                        //Initialiazing data
                        $scope.dataxPump = [], $scope.datayPump = [], $scope.datax[0] = [], $scope.datay[0] = [], $scope.datax[1] = [], $scope.datay[1] = [], $scope.datax[2] = [], $scope.datay[2] = [];
                        $scope.avgMagnetism[0] = [], $scope.avgMagnetism[1] = [], $scope.avgMagnetism[2] = [];
                        $scope.isVa[0] = [], $scope.isVa[1] = [], $scope.isVa[2] = [];
                        $scope.isFe[0] = [], $scope.isFe[1] = [], $scope.isFe[2] = [];
                        $scope.isPb[0] = [], $scope.isPb[1] = [], $scope.isPb[2] = [];
                        $scope.isCu[0] = [], $scope.isCu[1] = [], $scope.isCu[2] = [];
                        $scope.isSn[0] = [], $scope.isSn[1] = [], $scope.isSn[2] = [];
                        $scope.isTurbi[0] = [], $scope.isTurbi[1] = [], $scope.isTurbi[2] = [];
                        $scope.avgTurbi[0] = [], $scope.avgTurbi[1] = [], $scope.avgTurbi[2] = [];
                        $scope.isMovie[0] = [], $scope.isMovie[1] = [], $scope.isMovie[2] = [];
                        $scope.dates[0] = [], $scope.dates[1] = [], $scope.dates[2] = [];

                        const viewUrl = "_design/by_date/_view/allData";

                        //Database key
                        const queryOptions = {
                            startkey: [dataStartYear, dataStartMonth, dataStartDay, 0, 0, 0, 1],
                            endkey: [dataEndYear, dataEndMonth, dataEndDay, 23, 59, 59, 3]
                        };
                        console.log(queryOptions)
                        //Searching in the database
                        couchBubblot1.get("bubblot", viewUrl, queryOptions).then(({ data, headers, status }) => {
                            //Loop for running through each data received
                            for (var i = 0; i < data.rows.length; i++) {
                                //Get from which bubblot comes the data
                                var bubblotIndex = data.rows[i].key[6] - 1;
                                //Get the date of the measurement
                                $scope.dates[bubblotIndex].push([data.rows[i].key[0], data.rows[i].key[1], data.rows[i].key[2], data.rows[i].key[3], data.rows[i].key[4], data.rows[i].key[5]]);
                                if (i == 0) {
                                    $scope.dataxPump.push(500);
                                    $scope.datayPump.push(600);
                                }
                                //Get x y coordinates
                                $scope.datax[bubblotIndex].push(parseInt($scope.dataxPump) + parseInt(data.rows[i].value.distancexToPump));
                                $scope.datay[bubblotIndex].push(parseInt($scope.datayPump) + parseInt(data.rows[i].value.distanceyToPump));
                                //Get magnetism
                                $scope.avgMagnetism[bubblotIndex].push(parseFloat(data.rows[i].value.magnetism.mean));
                                //Get if there is VA
                                if (data.rows[i].value.VA.available) $scope.isVa[bubblotIndex].push(true);
                                else $scope.isVa[bubblotIndex].push(false);
                                //Get if there is Fe
                                if (data.rows[i].value.VA.fe) $scope.isFe[bubblotIndex].push(true);
                                else $scope.isFe[bubblotIndex].push(false);
                                //Get if there is Pb
                                if (data.rows[i].value.VA.pb) $scope.isPb[bubblotIndex].push(true);
                                else $scope.isPb[bubblotIndex].push(false);
                                //Get if there is Cu
                                if (data.rows[i].value.VA.cu) $scope.isCu[bubblotIndex].push(true);
                                else $scope.isCu[bubblotIndex].push(false);
                                //Get if there is Sn
                                if (data.rows[i].value.VA.sn) $scope.isSn[bubblotIndex].push(true);
                                else $scope.isSn[bubblotIndex].push(false);
                                //Get if there is turbidity
                                if (data.rows[i].value.turbidity.available) {
                                    $scope.isTurbi[bubblotIndex].push(true);
                                    //Get turbidity average values for each color
                                    $scope.avgTurbi[bubblotIndex].push([data.rows[i].value.turbidity.avgRed, data.rows[i].value.turbidity.avgGreen, data.rows[i].value.turbidity.avgBlue]);
                                }
                                else {
                                    $scope.isTurbi[bubblotIndex].push(false);
                                    $scope.avgTurbi[bubblotIndex].push([null, null, null]);
                                }
                                //Get if there is a movie
                                if (data.rows[i].value.movie.available) $scope.isMovie[bubblotIndex].push(true);
                                else $scope.isMovie[bubblotIndex].push(false);
                            }
                            $scope.extractingData = !$scope.extractingData;
                            $scope.$apply();
                            // data is json response 
                            // headers is an object with all response headers 
                            // status is statusCode number 
                        }, err => {
                            console.log(err.code);
                            // either request error occured 
                            // ...or err.code=EDOCMISSING if document is missing 
                            // ...or err.code=EUNKNOWN if statusCode is unexpected 
                        });
                    })
                    return number;
                }
                var days = document.createElement('ul');
                days.className += "cld-days";
                // Previous Month's Days
                for (var i = 0; i < (objCalendar.Selected.FirstDay); i++) {
                    var day = document.createElement('li');
                    day.className += "cld-day prevMonth";
                    //Disabled Days
                    var d = i % 7;
                    for (var q = 0; q < objCalendar.Options.DisabledDays.length; q++) {
                        if (d == objCalendar.Options.DisabledDays[q]) {
                            day.className += " disableDay";
                        }
                    }

                    var number = DayNumber((objCalendar.Prev.Days - objCalendar.Selected.FirstDay) + (i + 1));
                    day.appendChild(number);

                    days.appendChild(day);
                }
                
                // Current Month's Days
                for (var i = 0; i < objCalendar.Selected.Days; i++) {
                    var day = document.createElement('li');
                    day.className += "cld-day currMonth";
                    //Disabled Days
                    var d = (i + objCalendar.Selected.FirstDay) % 7;
                    for (var q = 0; q < objCalendar.Options.DisabledDays.length; q++) {
                        if (d == objCalendar.Options.DisabledDays[q]) {
                            day.className += " disableDay";
                        }
                    }
                    var number = DayNumber(i + 1);
                    // Check Date against Event Dates
                    for (var n = 0; n < objCalendar.Model.length; n++) {
                        var evDate = objCalendar.Model[n].Date;
                        var toDate = new Date(objCalendar.Selected.Year, objCalendar.Selected.Month, (i + 1));
                        if (evDate.getTime() == toDate.getTime()) {
                            number.className += " eventday";
                            var title = document.createElement('span');
                            title.className += "cld-title";
                            if (typeof objCalendar.Model[n].Link == 'function' || objCalendar.Options.EventClick) {
                                var a = document.createElement('a');
                                a.setAttribute('href', '#');
                                a.innerHTML += objCalendar.Model[n].Title;
                                if (objCalendar.Options.EventClick) {
                                    var z = objCalendar.Model[n].Link;
                                    if (typeof objCalendar.Model[n].Link != 'string') {
                                        a.addEventListener('click', objCalendar.Options.EventClick.bind.apply(objCalendar.Options.EventClick, [null].concat(z)));
                                        if (objCalendar.Options.EventTargetWholeDay) {
                                            day.className += " clickable";
                                            day.addEventListener('click', objCalendar.Options.EventClick.bind.apply(objCalendar.Options.EventClick, [null].concat(z)));
                                        }
                                    } else {
                                        a.addEventListener('click', objCalendar.Options.EventClick.bind(null, z));
                                        if (objCalendar.Options.EventTargetWholeDay) {
                                            day.className += " clickable";
                                            day.addEventListener('click', objCalendar.Options.EventClick.bind(null, z));
                                        }
                                    }
                                } else {
                                    a.addEventListener('click', objCalendar.Model[n].Link);
                                    if (objCalendar.Options.EventTargetWholeDay) {
                                        day.className += " clickable";
                                        day.addEventListener('click', objCalendar.Model[n].Link);
                                    }
                                }
                                title.appendChild(a);
                            } else {
                                title.innerHTML += '<a href="' + objCalendar.Model[n].Link + '">' + objCalendar.Model[n].Title + '</a>';
                            }
                            number.appendChild(title);
                        }
                    }
                    day.appendChild(number);
                    // If Today..
                    if ((i + 1) == objCalendar.Today.getDate() && objCalendar.Selected.Month == objCalendar.Today.Month && objCalendar.Selected.Year == objCalendar.Today.Year) {
                        day.className += " today";
                    }
                    for(var j=0; j<dataDate.length; j++){
                        if((i+1) == dataDate[j][2] && objCalendar.Selected.Month + 1 == dataDate[j][1] && objCalendar.Selected.Year == dataDate[j][0]){
                            day.className += " data";
                        }
                    }
                    days.appendChild(day);
                }
                // Next Month's Days
                // Always same amount of days in calander
                var extraDays = 13;
                if (days.children.length > 35) { extraDays = 6; }
                else if (days.children.length < 29) { extraDays = 20; }

                for (var i = 0; i < (extraDays - objCalendar.Selected.LastDay); i++) {
                    var day = document.createElement('li');
                    day.className += "cld-day nextMonth";
                    //Disabled Days
                    var d = (i + objCalendar.Selected.LastDay + 1) % 7;
                    for (var q = 0; q < objCalendar.Options.DisabledDays.length; q++) {
                        if (d == objCalendar.Options.DisabledDays[q]) {
                            day.className += " disableDay";
                        }
                    }

                    var number = DayNumber(i + 1);
                    day.appendChild(number);
 
                    days.appendChild(day);
                }
                mainSection.appendChild(days);
            }
            if (objCalendar.Options.Color) {
                mainSection.innerHTML += '<style>.cld-main{color:' + objCalendar.Options.Color + ';}</style>';
            }
            if (objCalendar.Options.LinkColor) {
                mainSection.innerHTML += '<style>.cld-title a{color:' + objCalendar.Options.LinkColor + ';}</style>';
            }
            element.appendChild(mainSection);

            if (objCalendar.Options.NavShow && objCalendar.Options.NavVertical) {
                AddSidebar();
            }
            if (objCalendar.Options.DateTimeShow) {
                AddDateTime();
            }
            AddLabels();
            AddDays();
        }
        var settings = {
            Color: '',
            LinkColor: '',
            NavShow: true,
            NavVertical: false,
            NavLocation: '',
            DateTimeShow: true,
            DateTimeFormat: 'mmm, yyyy',
            DatetimeLocation: '',
            EventClick: '',
            EventTargetWholeDay: false,
            DisabledDays: [],
            ModelChange: element
        };
        var objCalendar=null;
        var Calendar = function (model, options, date) {
            // Default Values
            this.Options = {
                Color: '',
                LinkColor: '',
                NavShow: true,
                NavVertical: false,
                NavLocation: '',
                DateTimeShow: true,
                DateTimeFormat: 'mmm, yyyy',
                DatetimeLocation: '',
                EventClick: '',
                EventTargetWholeDay: false,
                DisabledDays: [],
                ModelChange: model
            };
            // Overwriting default values
            for (var key in options) {
                this.Options[key] = typeof options[key] == 'string' ? options[key].toLowerCase() : options[key];
            }

            model ? this.Model = model : this.Model = {};
            this.Today = new Date(2022,5);

            this.Selected = this.Today
            this.Today.Month = this.Today.getMonth();
            this.Today.Year = this.Today.getFullYear();
            if (date) { this.Selected = date }
            this.Selected.Month = this.Selected.getMonth();
            this.Selected.Year = this.Selected.getFullYear();

            this.Selected.Days = new Date(this.Selected.Year, (this.Selected.Month + 1), 0).getDate();
            this.Selected.FirstDay = new Date(this.Selected.Year, (this.Selected.Month), 1).getDay();
            this.Selected.LastDay = new Date(this.Selected.Year, (this.Selected.Month + 1), 0).getDay();

            this.Prev = new Date(this.Selected.Year, (this.Selected.Month - 1), 1);
            if (this.Selected.Month == 0) { this.Prev = new Date(this.Selected.Year - 1, 11, 1); }
            this.Prev.Days = new Date(this.Prev.getFullYear(), (this.Prev.getMonth() + 1), 0).getDate();
        };
        function caleandar(el, data, settings) {
            objCalendar = new Calendar(data, settings);
            createCalendar(el);
        }

        // Check if there is data
        var dataDate = new Array();
        const viewUrl2 = "_design/by_date/_view/key";
        //Searching in the database
        couchBubblot1.get("bubblot", viewUrl2).then(({ data, headers, status }) => {
            for(var i = 0; i<data.rows.length; i++){
                if(dataDate.length>0){
                    for(var j = 0; j<dataDate.length; j++){
                        if(data.rows[i].value[0] != dataDate[j][0] || data.rows[i].value[1] != dataDate[j][1] || data.rows[i].value[2] != dataDate[j][2]){
                            dataDate.push([data.rows[i].value[0], data.rows[i].value[1],data.rows[i].value[2]])
                        }
                    }
                }
                else dataDate.push([data.rows[i].value[0], data.rows[i].value[1],data.rows[i].value[2]])
            }
            caleandar(element);
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

}());
