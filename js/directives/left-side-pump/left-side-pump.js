(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('leftSidePump', leftPumpDirective);

    function leftPumpDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/left-side-pump/left-side-pump.html',
            controller: 'leftSidePumpCtrl'
        }
    }

}());
