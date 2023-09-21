(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('direction', directionDirective);

    function directionDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/direction/direction.html',
            replace: true,
            controller: 'directionCtrl',
            controllerAs: 'vm',
            scope: { angleStorage: '=', angleNorth:'=', angleB1: '=', angleB2: '=', angleB3: '='}
        }
    }

}());
