(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('message', messageDirective);

    function messageDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/message/message.html',
            replace: true,
            controller: 'messageCtrl',
            scope: {displayListEmo:'='},
            link: function(scope, element, attr) {
                scope.$watch('displayListEmo', function (value) {
                    scope.displayListEmoFunc(value);
                });
            }
        }
    }
}());
