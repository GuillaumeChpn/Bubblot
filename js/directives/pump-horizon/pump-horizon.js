(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('pumpHorizon', pumpHorizonDirective);

    function pumpHorizonDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/pump-horizon/pump-horizon.html',
            replace: true,
            controller: 'pumpHorizonCtrl',
            controllerAs: 'vm',
            scope: {pitch: '=', roll: '='}
        }
    }

}());
