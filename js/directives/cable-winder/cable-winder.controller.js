(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('cableWinderCtrl', ['$timeout', '$scope', cableWinderCtrl]);

    function cableWinderCtrl($timeout,$scope) {
        var vm = this;
        $scope.panelSize = 280;
        $scope.panelThickness = 30;
    }

}());
