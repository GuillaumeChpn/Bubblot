(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('thrust', thrustDirective);

    function thrustDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/thrust/thrust.html',
            replace: true,
            controller: 'thrustCtrl',
            controllerAs: 'vm',
            scope: { engine1Angle: '=', engine1Radius: '=', engine2Angle: '=', engine2Radius: '=',
                    engine3Angle: '=', engine3Radius: '=', engine4Angle: '=', engine4Radius: '=',
                    circleSize: '=', circleThickness: '=', thrust: '=', thrustTopSwitchOn: '=', 
                    thrustBottomSwitchOn: '=', thrustDragOn: '=', help:'=',focusIndex: '='}
        }
    }

} ());
