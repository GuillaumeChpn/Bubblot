(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('leftSideCtrl', ['$timeout', '$scope', leftSideCtrl]);

    function leftSideCtrl($timeout, $scope) {
        var vm = this,
            focusLeftHandler = null;

        $scope.nozeSize = 145;
        $scope.nozeThickness = 25;
        $scope.storageSize = 145;
        $scope.storageThickness = 25;

        $scope.focusLeft = function (index, completeHandler) {
            if ($scope.leftData.focusLeftIndex == index) return;
            $scope.leftData.positionClass[$scope.leftData.focusLeftIndex] = $scope.leftData.positionClass[$scope.leftData.focusLeftIndex] + " transition-left";
            var clickedItemClass = $scope.leftData.positionClass[index];
            $scope.leftData.positionClass[index] = $scope.leftData.positionClass[index] + " transition-left";
            $timeout(function () {
                $scope.leftData.positionClass[$scope.leftData.focusLeftIndex] = $scope.leftData.focusOrigClass;
                $scope.leftData.focusOrigClass = clickedItemClass;
                $scope.leftData.positionClass[index] = "focus-left";
                $scope.leftData.focusLeftIndex = index;
                if (focusLeftHandler) {
                    focusLeftHandler(false);
                }
                if (completeHandler) {
                    completeHandler(true);
                    focusLeftHandler = completeHandler;
                }
            }, 300);
        };

        $scope.setNozeSize = function (focus) {
            if (focus) {
                $scope.nozeSize = 280;
                $scope.nozeThickness = 30;
            } else {
                $scope.nozeSize = 145;
                $scope.nozeThickness = 25;
            }
            $scope.$apply();
        };
        $scope.setStorageSize = function (focus) {
            if (focus) {
                $scope.storageSize = 280;
                $scope.storageThickness = 30;
            } else {
                $scope.storageSize = 145;
                $scope.storageThickness = 25;
            }
            $scope.$apply();
        };
    }

} ());
