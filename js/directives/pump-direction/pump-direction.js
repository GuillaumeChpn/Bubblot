(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('pumpDirection', pumpDirectionDirective);

    function pumpDirectionDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/pump-direction/pump-direction.html',
            replace: true,
            controller: 'pumpDirectionCtrl',
            controllerAs: 'vm',
            scope: { angleStorage: '=', angleNorth:'=', angleB1: '=', angleB2: '=', angleB3: '=', help: '=', focusIndex: '=', isCtrl: '='}
        }
    }

}());
