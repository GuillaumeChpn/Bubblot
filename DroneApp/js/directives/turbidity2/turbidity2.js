(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('turbidity2', turbidity2Directive);

    function turbidity2Directive() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/turbidity2/turbidity2.html',
            replace: true,
            controller: 'turbidity2Ctrl',
            scope: { turbidityRed: '=', turbidityGreen: '=', turbidityBlue: '='} ,
            
        }
    }

}());
