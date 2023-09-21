(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('map', mapDirective);

    function mapDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/map/map.html',
            controller: 'mapCtrl'  
        } 
    }
}());
