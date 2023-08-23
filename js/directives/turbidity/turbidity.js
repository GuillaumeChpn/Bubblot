(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('turbidity', turbidityDirective);

    function turbidityDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/turbidity/turbidity.html',
            replace: true,
            controller: 'turbidityCtrl',
            scope: { turbidityRed: '=', turbidityGreen: '=', turbidityBlue: '='} ,
            
        }
    }

}());

