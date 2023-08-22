(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('cableWinder', cableWinderDirective);

    function cableWinderDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/cable-winder/cable-winder.html',
            controller: 'cableWinderCtrl'
        }
    }

}());
