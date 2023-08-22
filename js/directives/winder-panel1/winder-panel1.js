(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('winderPanel1', winderPanel1Directive);

    function winderPanel1Directive() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/winder-panel1/winder-panel1.html',
            replace: true,
            controller: 'winderPanel1Ctrl',
            controllerAs: 'vm',
            scope: {winderLength: '=', winderSpeed: '=', circleSize: '=', circleThickness: '=', pressure: '=', minPressure: '=', pressureAlert: '=', 
                    reset: '=', railMode: '=', winderAlert:'='},
            link: function(scope, element, attr) {
                scope.$watch('winderLength', function (value) {
                    scope.updateLength(value);
                });        
                scope.$watch('pressure', function (value) {
                    scope.checkPressure(value);
                });     
                scope.$watch('minPressure', function (value) {
                    scope.checkPressure(value);
                }); 
            }
        }
    }

}());
