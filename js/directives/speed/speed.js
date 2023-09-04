(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('speed', speedDirective);

    function speedDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/speed/speed.html',
            replace: true,
            controller: 'speedCtrl',
            controllerAs: 'vm',
            scope: {circleSize: '=', circleThickness: '=', thrust: '=', thrustTopSwitchOn: '=', 
                    thrustBottomSwitchOn: '=', thrustDragOn: '=', help:'=',focusIndex: '=',
                    cameraOdomMsg:"=", cameraOdomArray: "=", cameraOdomTime: "=", cameraOdomAngle: "="}
        }
    }

} ());
