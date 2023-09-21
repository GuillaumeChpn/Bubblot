(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('pumpAlert', pumpAlertDirective);

    function pumpAlertDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/pump-alert/pump-alert.html',
            controller: 'pumpAlertCtrl',
            scope: {security: '=', audio: '='},
            link: function(scope, element, attr) {
                scope.$watch('security', function (value) {
                    scope.alert();
                });    
            } 
        }
    }
}());
