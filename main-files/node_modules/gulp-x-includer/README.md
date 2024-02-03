# gulp-x-includer

Gulp plugin for include js/css/html... any file.

### Install

```shell
$ npm install gulp-x-includer --save-dev
```

### Syntax

Default supported 4 comment syntax.

```
'' or ""
<!-- include "path/to/xxx.html" -->
// include "path/to/xxx.js"
/* include "path/to/xxx.css" */
# include "./include/test.txt"
```

HTML :

```html
<!-- include "path/to/xxx.html" -->
```

JavaScript :

```javascript
// include "path/to/xxx.js"
```

CSS :

```css
/* include "path/to/xxx.css" */
```

Other file (e.g. markdown .md, configure .conf, .ini ...) :

```markdown
# include "./include/test.txt"
```

### Examples

`test.html` :

```html
<!DOCTYPE html>
<html>
<head>
    <title>Test HTML Document</title>
    <meta charset="UTF-8">
    <!-- fdsfsadf -->
    <meta name="description" content="">
    <meta name="keywords" content="">
    <style>
        /* include "./css/test.css" */
        /* fasdfadsfsadf */
        div {color: red;}
        /* fasdfadsfsadf */
        /* include "./css/test1.css" */
        p {color: green;}
        /* fasdfadsfsadf */
        strong {color: yellow;}
        /* fasdfadsfsadf */
        /* include "./css/test2.css" */
    </style>
</head>
<body>
    <!-- include "./html/header.html" -->
    <!-- fdsfsadf -->
    <!-- test -->
    <p>TEST</p>
    <!-- include "./html/main.html" -->
    <!-- fdsfsadf -->
    <div>TEST</div>
    <!-- include "./html/footer.html" -->
    <!-- fdsfsadf -->
    
    <script>
        // include './js/test.js'
        var a = 234567890, b, c;
        // fadsfasdfdsafsdfsd
        // fadsfasdfdsafsdfsd
        // fadsfasdfdsafsdfsd
        // fadsfasdfdsafsdfsd
        // include './js/functions.js'

        $(function(){
            alert(345678);
        });
    </script>
</body>
</html>
```

`Gulpfile.js` :

```javascript
var gulp         = require('gulp');
var includer     = require("gulp-x-includer");

gulp.task("include", function(){
    gulp.src(["./*.html", "./*.css", "./*.js"])
    .pipe(includer())
    .pipe(gulp.dest("./dist"));
});
```

Build :

```shell
$ gulp include
```

### Options

```javascript
{
    debug        : true | false,         // print file includes for console.log(), default false.
    debugOptions : true | false,         // print options object for console.log(), default false.
    regexs       : {},                   // define you include regex
    transform    : function(content) {}  // transform include file content
}
```

### Changes

[Change logs](https://github.com/pandao/gulp-x-includer/blob/master/CHANGE.md)

### License

The [MIT](https://github.com/pandao/gulp-x-includer/blob/master/LICENSE) License.

Copyright (c) 2015 Pandao