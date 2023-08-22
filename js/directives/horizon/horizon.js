(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('horizon', horizonDirective);

    function horizonDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/horizon/horizon.html',
            replace: true,
            controller: 'horizonCtrl',
            controllerAs: 'vm',
            scope: {pitch: '=', roll: '=', twistLeft: '=', twistRight: '='},
        }
    }

}());
