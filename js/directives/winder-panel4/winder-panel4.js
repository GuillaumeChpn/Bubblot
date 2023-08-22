(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('winderPanel4', winderPanel4Directive);

    function winderPanel4Directive() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/winder-panel4/winder-panel4.html',
            replace: true,
            controller: 'winderPanel4Ctrl',
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
