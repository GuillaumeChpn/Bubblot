(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('draggable', ['$document', draggableDirective]);

    function draggableDirective($document) {
        return {
            scope: {position: '='}, 
            link: function (scope, element, attr) {
                var startX = 0, x=0, clientRect;
                element.on('mousedown', function (event) {
                    var cadran=document.getElementsByClassName('spotlight-progressbar');
                    clientRect = cadran[0].getBoundingClientRect();
                    event.preventDefault();
                    element.addClass('on');
                    x = clientRect.width * (scope.position/100);
                    startX = event.screenX - x;
                    $document.on('mousemove', mousemove);
                    $document.on('mouseup', mouseup);
                });

                function mousemove(event) {
                        x = event.screenX - startX;
                        if (element.hasClass('on')) {
                            if (x >= 0 && x <= clientRect.width) {
                                scope.position = (x / clientRect.width) * 100;
                                scope.$apply();
                                element.css({
                                    left: scope.position + '%'
                                });
                            }
                        }
                        else {
                            element.css({
                                left: 0
                            });
                        }
                }

                function mouseup() {
                    $document.off('mousemove', mousemove);
                    $document.off('mouseup', mouseup);
                }
            }
        }
    }
} ());
