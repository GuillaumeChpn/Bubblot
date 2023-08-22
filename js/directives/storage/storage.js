(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('storage', storageDirective);

    function storageDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/storage/storage.html',
            replace: true,
            controller: 'storageCtrl',
            controllerAs: 'vm',
            scope: {pumpOn: '=', pumpPower:'=', filledOk: '=', circleSize: '=', circleThickness: '=',focusIndex: '=' }
        }
    }

}());
