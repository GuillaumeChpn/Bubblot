(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('rightSidePump', rightPumpDirective);

    function rightPumpDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/right-side-pump/right-side-pump.html',
            controller: 'rightSidePumpCtrl'
        }
    }

} ());
