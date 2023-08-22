(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('circleProgress', circleProgress);

    /**@ngInject*/
    function circleProgress($parse) {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/circle-progress/circle-progress.html',
            scope: { circleSize: '=', circleThickness: '=', value: '=', reverse: '=', circleRatio: '=', circleColor: '=' },
            replace: true,
            link: function (scope, element, attr) {
                var createCircle = function () {
                    var color;
                    switch (scope.circleColor) {
                        case 0:
                            color = ["#c3f900"];
                            break;
                        case 1:
                            color = ["#FF0000"];
                            break;
                        case 2:
                            color = ["#0000FF"];
                            break;
                    }
                    $(element).circleProgress({
                        value: scope.value * scope.circleRatio,
                        size: scope.circleSize,
                        reverse: scope.reverse ? true : false,
                        emptyFill: "rgba(0, 0, 0, 0)",
                        thickness: scope.circleThickness,
                        fill: {
                            gradient: color
                        }
                    });
                };
                createCircle();
                var oldValue = scope.value;
                scope.$watch('value', function (value) {
                    if (value != oldValue) {
                        $(element).circleProgress('value', scope.value * scope.circleRatio);
                        oldValue = value;
                    }
                });
                var oldCircleSize = scope.circleSize;
                scope.$watch('circleSize', function (size) {
                    if (size && oldCircleSize != size) {
                        createCircle();
                        oldCircleSize = size;
                    }
                });
                scope.$watch('circleColor', function (size) {
                    var color;
                    switch (scope.circleColor) {
                        case 0:
                            color = ["#c3f900"];
                            break;
                        case 1:
                            color = ["#FF0000"];
                            break;
                        case 2:
                            color = ["#0000FF"];
                            break;
                    }
                    $(element).circleProgress('fill', { gradient: color });
                    createCircle();
                });
            }
        };
    }
}());
