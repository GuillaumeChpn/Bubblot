(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('turbidity2Ctrl', turbidity2Ctrl);

    function turbidity2Ctrl($scope, $element) {
        var vm = this;
        var labels = ["Red", "Green", "Blue"];

        var barData=[$scope.turbidityRed, $scope.turbidityGreen, $scope.turbidityBlue];
        var initData = {
            labels: labels,
            datasets: [{
                backgroundColor: 'rgba(' + $scope.turbidityRed + ', ' + $scope.turbidityGreen + ', ' + $scope.turbidityBlue + ', ' + '0.8' +')',
                borderColor: 'rgb(' + $scope.turbidityRed + ', ' + $scope.turbidityGreen + ', ' + $scope.turbidityBlue +')',
                data: barData,
            }]
        };
        var context = $element.find("canvas")[0].getContext("2d");
        var graph = new Chart(context, {
            type: 'radar',
            data: initData,
            options: {
                maintainAspectRatio: false,
                legend: {
                    display: false,
                },
                tooltips: {
                    enabled: false,
                },
                scale: {
                    ticks: {
                        min: 0,
                        max: 100,
                        stepSize: 5,
                        showLabelBackdrop: false
                    }
                }
            }
        });
        //Update the information graph when clicking on map
        $scope.updateturbidity2 = function () {
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
