(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('leftSide', leftDirective);

    function leftDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/left-side/left-side.html',
            controller: 'leftSideCtrl'
        }
    }

}());
