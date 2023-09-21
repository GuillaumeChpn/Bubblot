(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('pumpDeep', pumpDeepDirective);

    function pumpDeepDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/pump-deep/pump-deep.html',
            replace: true,
            controller: 'pumpDeepCtrl',
            controllerAs: 'vm',
            scope: {security: '=', securityAlert: '=', ballastState: '=', circleSize: '=', circleThickness: '=', depth: '=', focusIndex: '='},
            link: function(scope, element, attr) {
                scope.$watch('security', function (value) {
                    scope.checkSecurity(value);
                });
            }
        }
    }

}());
