(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('camera', cameraDirective);

    function cameraDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/camera/camera.html',
            replace: true,
            controller: 'cameraCtrl',
            controllerAs: 'vm',
            scope: {cameraRec: '=', cameraRecMenu: '=', cameraPlay: '='}
        }
    }

}());
