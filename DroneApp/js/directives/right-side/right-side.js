(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('rightSide', rightDirective);

    function rightDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/right-side/right-side.html',
            controller: 'rightSideCtrl'
        }
    }

} ());
