const { src, dest, series, watch } = require('gulp');
const run = require('gulp-run-command').default;

// Gulp Sass
const sass = require('gulp-sass')(require('sass'));
const sourcemaps = require('gulp-sourcemaps');
var includer = require("gulp-x-includer");

const minify = require('gulp-minifier');

//////////////////////////////////////

// COMPILE - HTML FILES
function htmls(cb) {
    src('src/html/**')
    .pipe(dest('dist'));
    cb();
}

// COMPILE - SCSS STYLE
function stylecss(cb) {
    src([`src/scss/*.scss`])
        .pipe(sourcemaps.init())
        .pipe(sass({outputStyle: 'compressed'}).on('error', sass.logError))
        .pipe(sourcemaps.write('./'))
        .pipe(dest(`dist/assets/css`))

    cb();
}

// COMPILE - JAVASCRIPTS
function jsvendor(cb) {

    src([`src/js/**`, `!src/js/bundle.js`, `!src/js/custom/**`])
        .pipe(dest(`dist/assets/js`));

    src([`src/js/bundle.js`])
        .pipe(includer())
        .pipe(dest(`dist/assets/js`));

    cb();
}

// COPYING - ASSETS & IMAGES
function assets(cb) {
    src([`src/assets/**`]).pipe(dest(`dist/assets`));
    src(`src/images/**`).pipe(dest(`dist/images`));
    cb();
}

// EXPORTS COMMAND FOR COMPILE
//////////////////////////////////////

exports.build = series(htmls, jsvendor, stylecss, assets);

exports.develop = function () {
    watch([`src/js/**`]).on('change', series(jsvendor))
    watch([`src/scss/**`], { ignoreInitial: false }, series(stylecss))
    watch([`src/html/*.html`]).on('change', series(htmls))
    watch([`src/assets/**`, `src/images/**`], series(assets))
}

exports.watch = run(['gulp build', 'gulp develop']);
