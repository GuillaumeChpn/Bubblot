(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('mapPanel', mapPanelDirective);

    function mapPanelDirective() {
        return {
            restrict: 'E',
            templateUrl: 'js/directives/map-panel/map-panel.html',
            controller: 'mapPanelCtrl',
            controllerAs: 'vm',
            scope: {extractingData: '=', displayPath: '=', displayVa: '=', displayTurbi: '=', displayMovie: '=',displayFe: '=', displayPb: '=', displayCu: '=', displaySn: '=', isVa: '=', isFe: '=', isPb: '=', 
                    infoVa: '=', isCu: '=', isSn: '=', avgMagnetism: '=', infoMagnetism: '=', dataxPump:'=', datayPump: '=', datax:'=', datay: '=' ,  displayMagn: '=', 
                    isTurbi: '=', isMovie: '=', zoom: '=', ximage: '=', yimage: '=', xnew: '=', ynew: '=', vaCursor: '=', turbiCursor: '=', magnCursor: '=', movieCursor: '=', avgTurbi: '=', updatePanel:'=',
                    dates: '=', dateCursor:'=', latitudeCursor: '=', longitudeCursor:'=', depthCursor:'=', tempCursor:'=', infoTurbiRed: '=', infoTurbiGreen: '=', infoTurbiBlue: '=', xlast:'=', ylast:'=', 
                    bubblotCursor: '=', bubblots: '=', playData: '=', calendarFwd:'=', calendarRwd:'='},
            link: function(scope, element, attr) {
                scope.$watch('updatePanel', function () {
                    scope.updateInfoPanel();
                });
                scope.$watch('calendarFwd', function () {
                    scope.updateCalendarFwd();
                });
                scope.$watch('calendarRwd', function () {
                    scope.updateCalendarRwd();
                });
            }
        } 
    }
}());
