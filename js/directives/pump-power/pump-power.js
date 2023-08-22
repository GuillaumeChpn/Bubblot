(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('pumpPower', pumpPowerDirective);

    function pumpPowerDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/pump-power/pump-power.html',
            replace: true,
            controller: 'pumpPowerCtrl',
            controllerAs: 'vm',
            scope: {needleAngle: '=',circleSize: '=', circleThickness: '=', thrust: '=', help: '=', focusIndex: '='}
        }
    }

}());
