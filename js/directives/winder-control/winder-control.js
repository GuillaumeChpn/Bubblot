(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('winderControl', winderControlDirective);

    function winderControlDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/winder-control/winder-control.html',
            replace: true,
            controller: 'winderControlCtrl',
            controllerAs: 'vm',
            scope: { mainControl: '=', winderControl1: '=', winderControl2: '=', winderControl3: '=', winderControl4: '=', railMode: '=', help: '=', railLength: '=' },
            link: function (scope, element, attr) {
                scope.$watch('railLength', function (value) {
                    scope.updateLength(value);
                });
            }
        }
    }

}());
