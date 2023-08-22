(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('bubblotAlert', bubblotAlertDirective);

    function bubblotAlertDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/bubblot-alert/bubblot-alert.html',
            controller: 'bubblotAlertCtrl',
            scope: {security: '=', twistLeft: '=', twistRight: '=', audio: '='},
            link: function(scope, element, attr) {
                scope.$watch('security', function (value) {
                    scope.alert();
                });    
                scope.$watch('twistLeft', function (value) {
                    scope.alert();
                });  
                scope.$watch('twistRight', function (value) {
                    scope.alert();
                });  
            } 
        }
    }
}());
