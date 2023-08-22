(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('pumpGps', pumpGpsDirective);

    function pumpGpsDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/pump-gps/pump-gps.html',
            replace: true,
            controller: 'pumpGpsCtrl',
            controllerAs: 'vm',
            scope: {compass: '=', localLong: '=', localLat: '=', targetLong: '=', targetLat: '='}
        }
    }

}());
