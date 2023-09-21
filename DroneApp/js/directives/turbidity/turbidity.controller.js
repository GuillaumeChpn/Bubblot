(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('turbidityCtrl', turbidityCtrl);

    function turbidityCtrl($scope, $element) {
        var vm = this;
        var labels = ["Red", "Green", "Blue"];

        var barData=[$scope.turbidityRed, $scope.turbidityGreen, $scope.turbidityBlue];
        
        var initData = {
            labels: labels,
            datasets: [{
                data: barData,
                borderWidth: 2,
                borderRadius: 50,
                backgroundColor: [
                    'rgba(255, 0, 0, 0.2)',
                    'rgba(0, 255, 0, 0.2)',
                    'rgba(0, 0, 255, 0.2)'],
                borderColor: [
                    'rgb(255, 0, 0)',
                    'rgb(0, 255, 0)',
                    'rgb(0, 0, 255)']
                }]
        };
        var context = $element.find("canvas")[0].getContext("2d");
        var graph = new Chart(context, {
            type: 'bar',
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
                    yAxes: [{
                        type: 'linear',
                        position: 'left',
                        ticks: {
                            min: 0,
                            max: 100,
                            stepSize: 25,
                            fontColor: "rgba(0, 0, 0, 0.6)"
                        },
                        gridLines: {
                            display: true,
                            color: "rgba(0, 0, 0, 0.6)",
                        }
                    }]
                }
            }
        });
        //Update the information graph when clicking on map
        $scope.updateturbidity = function () {
            /*
            $scope.data = [];
            for (var i = 0; i <= 10; i++) {
                //var obj = { x: 0.2 * i, y: 100 * Math.random() };
                var obj = 100 * Math.random();
                $scope.data.push(obj);
            }
            graph.data.datasets[1].data = $scope.data;
            graph.update();
            */
        };
    }
}());
