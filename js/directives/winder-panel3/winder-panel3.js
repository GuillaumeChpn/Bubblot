(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('winderPanel3', winderPanel3Directive);

    function winderPanel3Directive() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/winder-panel3/winder-panel3.html',
            replace: true,
            controller: 'winderPanel3Ctrl',
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
