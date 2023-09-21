(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('lighting', lightingDirective);

    function lightingDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/lighting/lighting.html',
            replace: true,
            controller: 'lightingCtrl',
            scope: {
                spotlightIntensity: '=',
                foglightIntensity: '=',
                spotlightSwitchOn: '=',
                foglightSwitchOn: '=',
                help:'=',
                focusIndex: '='
            }
        }
    }

}());
