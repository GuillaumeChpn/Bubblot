(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('winderPanel2', winderPanel2Directive);

    function winderPanel2Directive() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/winder-panel2/winder-panel2.html',
            replace: true,
            controller: 'winderPanel2Ctrl',
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
