(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('noze', nozeDirective);

    function nozeDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/noze/noze.html',
            replace: true,
            controller: 'nozeCtrl',
            controllerAs: 'vm',
            scope: {
                circleSize: '=', circleThickness: '=', turbidityRed: '=',turbidityGreen: '=',turbidityBlue: '=',magnetism: '=', magnetismXaxis:'=', magnetismYaxis:'='
            },
            link: function(scope, element, attr) {  
                scope.$watch('magnetismXaxis', function () {
                    scope.updateMagnetism();
                });  
                scope.$watch('magnetismYaxis', function () {
                    scope.updateMagnetism();
                });   
            }
        }
    }
} ());
