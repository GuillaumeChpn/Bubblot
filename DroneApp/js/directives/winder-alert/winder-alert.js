(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('winderAlert', winderAlertDirective);

    function winderAlertDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/winder-alert/winder-alert.html',
            controller: 'winderAlertCtrl',
            scope: {pressure1: '=', reset1: '=', pressure2: '=', reset2: '=', pressure3: '=', reset3: '=', pressure4: '=', reset4: '=', audio: '='},
            link: function(scope, element, attr) {
                scope.$watch('pressure1', function (value) {
                    scope.alert();
                });    
                scope.$watch('reset1', function (value) {
                    scope.alert();
                });
                scope.$watch('pressure2', function (value) {
                    scope.alert();
                });    
                scope.$watch('reset2', function (value) {
                    scope.alert();
                }); 
                scope.$watch('pressure3', function (value) {
                    scope.alert();
                });    
                scope.$watch('reset3', function (value) {
                    scope.alert();
                }); 
                scope.$watch('pressure4', function (value) {
                    scope.alert();
                });    
                scope.$watch('reset4', function (value) {
                    scope.alert();
                });  
                scope.$watch('rail', function (value) {
                    scope.alert();
                });  
            } 
        }
    }
}());
