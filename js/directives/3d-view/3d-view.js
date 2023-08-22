(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('threeDView', ThreeDViewDirective);

    /**@ngInject*/
    function ThreeDViewDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/3d-view/3d-view.html',
            replace: true,
            controller: '3dViewCtlr',
            controllerAs: 'vm',
            scope: { angle: '=', help: '=', focusIndex: '='}
        }
    }

}());
